import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import discretize
from simpeg import (
    Data, maps,
    data_misfit, regularization, optimization, inverse_problem,
    inversion, directives, utils
)
from simpeg.electromagnetics import resistivity as dc
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    apparent_resistivity_from_voltage,
    plot_pseudosection,
)

clim_diff_rhoa=0.5
clim_diff_misfit=0.6

def create_survey(
    survey_type="dipole-dipole",
    end_points=np.r_[-350.0, 450.0],
    station_spacing=50.0,
    num_rx_per_src=8,
):
    import warnings
    warnings.filterwarnings("ignore")
    data_type = "volt"
    source_list = generate_dcip_sources_line(
        survey_type=survey_type,
        data_type=data_type,
        dimension_type="2D",
        end_points=end_points,
        topo=0,
        num_rx_per_src=num_rx_per_src,
        station_spacing=station_spacing,
    )

    # Define survey
    survey = dc.survey.Survey(source_list, survey_type=survey_type)
    return survey


def build_mesh(
    survey,
    n_cells_per_spacing_x=4,
    n_cells_per_spacing_z=4,
    n_core_extra_x=4,
    n_core_extra_z=4,
    core_domain_z_ratio=1/3.,
    padding_factor=1.3,
    n_pad_x=10,
    n_pad_z=10,
):
    """
    A function for designing a Tensor Mesh based on DC survey parameters

    Parameters
    ----------

    survey: A DC (or IP) survey object

    n_cells_per_spacing_[x, z]:  Number of [x, z]-cells per the minimum electrode spacing

    n_core_extra_[x, z]: Number of extra cells with the same size as the core domain beyond the survey extent

    core_domain_z_ratio: Factor that multiplies the maximum AB, MN separation to define the core mesh extent

    padding_factor: Factor by which we expand the mesh cells in the padding region

    n_pad_[x, z]: Number of padding cells in the x, z directions
    """
    min_electrode_spacing = np.min(np.abs(survey.locations_a[:, 0] - survey.locations_b[:, 0]))

    dx = min_electrode_spacing / n_cells_per_spacing_x
    dz = min_electrode_spacing / n_cells_per_spacing_z

    # define the x core domain
    core_domain_x = np.r_[
        survey.locations_a[:, 0].min(),
        survey.locations_b[:, 0].max()
    ]

    # find the y core domain
    # find the maximum spacing between source, receiver midpoints
    mid_ab = (survey.locations_a + survey.locations_b)/2
    mid_mn = (survey.locations_m + survey.locations_n)/2
    separation_ab_mn = np.abs(mid_ab - mid_mn)
    max_separation = separation_ab_mn.max()
    core_domain_z = np.r_[-core_domain_z_ratio * max_separation, 0.]

    # add extra cells beyond the core domain
    n_core_x = np.ceil(np.diff(core_domain_x)/dx) + n_core_extra_x*2  # on each side
    n_core_z = np.ceil(np.diff(core_domain_z)/dz) + n_core_extra_z  # just below

    # define the tensors in each dimension
    hx = [(dx, n_pad_x, -padding_factor), (dx, n_core_x), (dx, n_pad_x, padding_factor)]
    hz = [(dz, n_pad_z, -padding_factor), (dz, n_core_z)]

    mesh = discretize.TensorMesh([hx, hz], x0="CN")

    return mesh, core_domain_x, core_domain_z

def plot_model(mesh, rho_plot, xlim=None, ylim=None, ax=None, clim=[None, None]):
    """
    Plot the model

    Parameters
    ----------

    mesh: the simulation mesh where the model is defined

    rho_plot: the model to be plotted

    xlim: x limits

    ylim: y limits

    ax: matplotlib axes object

    clim: colorbar limits

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    out = mesh.plot_image(
        rho_plot, ax=ax, pcolor_opts={"norm":LogNorm(vmin=clim[0], vmax=clim[1]), "cmap":"Spectral"}
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect(1)
    cb = plt.colorbar(out[0], ax=ax)
    cb.set_label("resistivity ($\Omega$m)")
    ax.set_ylabel("z (m)")
    ax.set_xlabel("x (m)")

    return ax


def plot_data(
    data_to_plot, data_type="apparent_resistivity",
    data_to_compare=None,
    ax=None, clim=None
):
    """
    Plot a pseudosection of the data
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 2.5))

    if data_type == "apparent_resistivity":
        cbar_label = "Apparent Resistivity ($\Omega$m)"
        dobs = None
    elif data_type == "misfit":
        cbar_label = "Misfit"
        dobs = data_to_plot.dobs - data_to_compare.dobs
    elif data_type == "normalized_misfit":
        cbar_label = "Normalized Misfit"
        dobs = (data_to_plot.dobs - data_to_compare.dobs)/data_to_compare.dobs
        clim = np.max(np.r_[clim.max(), np.max(np.abs(dobs))]) * np.r_[-1, 1]

    out = plot_pseudosection(
        data_to_plot, dobs=dobs, data_type=data_type, ax=ax,
        plot_type="contourf", data_locations=True,
        cbar_label=cbar_label,
        clim=clim,
        contourf_opts={"levels":10}
    )
    ax.set_yticklabels([])
    ax.set_ylabel("n spacing")

def plot_data_target_and_background(
    target_data, background_data, ax=None
):
    ax=None
    if ax is None:
        fig, ax = plt.subplots(3, 1, figsize=(8, 7))

    rhoa_target = apparent_resistivity_from_voltage(target_data.survey, target_data.dobs)
    rhoa_back = apparent_resistivity_from_voltage(background_data.survey, background_data.dobs)

    log_rhoa_center = np.log(np.mean(rhoa_back))
    clim_rho_a = np.max(np.r_[clim_diff_rhoa, 0.5*np.max(np.abs(np.log(rhoa_target)-log_rhoa_center))])
    clim_rho_a = np.exp(log_rhoa_center+clim_rho_a*np.r_[-1, 1])

    plot_data(target_data, data_type="apparent_resistivity", ax=ax[0], clim=clim_rho_a)
    ax[0].set_title("data with target")

    plot_data(background_data, data_type="apparent_resistivity", ax=ax[1], clim=clim_rho_a)
    ax[1].set_title("background data")

    plot_data(
        target_data, data_to_compare=background_data, data_type="normalized_misfit", ax=ax[2], clim=clim_diff_misfit*np.r_[-1, 1]
    )
    ax[2].set_title("normalized misfit")
    plt.tight_layout()

def create_inversion(
    simulation,
    synthetic_data,
    reference_resistivity,
    relative_error=0.05, noise_floor=1e-4,
    alpha_s=1e-4, alpha_x=1, alpha_y=1,
    maxIter=20, maxIterCG=20,
    beta0_ratio=1e1, cool_beta=True,
    beta_cooling_factor=2, beta_cooling_rate=1,
    use_target=True, chi_factor=1,
):
    inv_prob=create_inverse_problem(
        simulation,
        synthetic_data,
        reference_resistivity,
        relative_error=relative_error, noise_floor=noise_floor,
        alpha_s=alpha_s, alpha_x=alpha_x, alpha_y=alpha_y,
        maxIter=20, maxIterCG=20,
    )

    # set up our directives
    beta_est = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
    target = directives.TargetMisfit(chifact=chi_factor)
    save = directives.SaveOutputDictEveryIteration()

    directives_list = [beta_est, save]

    if use_target is True:
        directives_list.append(target)

    if cool_beta is True:
        beta_schedule = directives.BetaSchedule(coolingFactor=beta_cooling_factor, coolingRate=beta_cooling_rate)
        directives_list.append(beta_schedule)

    return inversion.BaseInversion(inv_prob, directiveList=directives_list), target, save

def create_inverse_problem(
    simulation,
    synthetic_data,
    reference_resistivity,
    relative_error=0.05, noise_floor=1e-4,
    alpha_s=1e-4, alpha_x=1, alpha_y=1,
    maxIter=20, maxIterCG=20,
):

    # set the uncertainties and define the data misfit
    synthetic_data.relative_error = relative_error
    synthetic_data.noise_floor = noise_floor
    dmisfit = data_misfit.L2DataMisfit(data=synthetic_data, simulation=simulation)

    mesh = simulation.mesh
    # regularization
    reg = regularization.WeightedLeastSquares(
        mesh, alpha_s=alpha_s, alpha_x=alpha_x, alpha_y=alpha_y, reference_model=np.log(reference_resistivity)*np.ones(mesh.n_cells)
    )

    # optimization
    opt = optimization.InexactGaussNewton(maxIter=maxIter, maxIterCG=maxIterCG)
    opt.remember("xc")

    # return the inverse problem
    return inverse_problem.BaseInvProblem(dmisfit, reg, opt)

def plot_inversion_results(
        simulation, synthetic_data, inversion_log, rho_background,
        core_domain_x=None, core_domain_z=None, iteration=None,
        ax=None, clim_model=None
        ):
    fig, ax = plt.subplots(2, 2, figsize=(12, 4))

    maxiter = len(inversion_log.outDict)
    iterations = np.arange(0, maxiter)
    betas = [inversion_log.outDict[i]["beta"] for i in range(1, maxiter+1)]
    phi_d = [inversion_log.outDict[i]["phi_d"] for i in range(1, maxiter+1)]
    phi_m = [inversion_log.outDict[i]["phi_m"] for i in range(1, maxiter+1)]

    if iteration is None:
        iteration=maxiter-1

    mesh = simulation.mesh
    survey = simulation.survey

    rho_recovered = np.exp(inversion_log.outDict[iteration]["m"])
    if clim_model is None:
        clim_factor = np.max(np.r_[5, np.abs(np.log10(rho_recovered)-np.log10(rho_background))])
        clim_model = rho_background*np.r_[1/clim_factor, clim_factor]

    plot_model(
        mesh, rho_recovered, core_domain_x, core_domain_z,
        ax=ax[0, 0], clim=clim_model
    )
    ax[0, 0].set_ylim(np.r_[-200, 0])
    ax[0, 0].set_title("predicted model")

    dpred = inversion_log.outDict[iteration]["dpred"]
    rhoa = apparent_resistivity_from_voltage(survey, dpred)

    log_rhoa_center = np.log(np.mean(rho_background))
    log_rhoa_max = 0.5*np.max(np.abs(np.log(rhoa)-log_rhoa_center))
    clim_rho_a = np.exp(log_rhoa_center+np.max(np.r_[clim_diff_rhoa, log_rhoa_max])*np.r_[-1, 1])

    out = plot_pseudosection(
        synthetic_data, dobs=rhoa, data_type="apparent_resistivity",
        plot_type="contourf", data_locations=True,
        cbar_label="Apparent Resistivity ($\Omega$m)",
        clim=clim_rho_a,
        contourf_opts={"levels":10},
        ax=ax[0, 1]
    )
    ax[0, 1].set_yticklabels([])
    ax[0, 1].set_ylabel("n spacing")
    ax[0, 1].set_title("predicted data")

    ax[1, 0].plot(iterations, phi_d, "-")
    ax[1, 0].plot(iterations, np.ones(len(betas))*survey.nD, "--k")
    ax[1, 0].set_xlabel("iteration")
    ax[1, 0].set_ylabel("$\\phi_d$", color="C0")
    ax[1, 0].set_title("Tikhonov curves")

    axm = ax[1, 0].twinx()
    axm.plot(iterations, phi_m, "-", color="C1")
    axm.set_ylabel("$\\phi_m$", color="C1")
    axm.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    if iteration is not None:
        ax[1,0].plot(iteration, phi_d[iteration], "ko")
        axm.plot(iteration, phi_m[iteration], "ko")

    misfit = (dpred-synthetic_data.dobs)/np.abs(synthetic_data.dobs)
    out = plot_pseudosection(
        synthetic_data, dobs=misfit, data_type="misfit",
        plot_type="contourf", data_locations=True, ax=ax[1,1],
        clim=np.max(np.r_[clim_diff_misfit, np.max(np.abs(misfit))])*np.r_[-1, 1]
    )
    ax[1, 1].set_yticklabels([])
    ax[1, 1].set_ylabel("n spacing")
    ax[1, 1].set_title("Normalized misfit")

    plt.tight_layout()
    return ax
