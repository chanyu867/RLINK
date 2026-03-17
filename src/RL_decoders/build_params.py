import os, glob
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def build_params(model_type, cfg: dict, input_dim, output_dim) -> dict:
    params = {}

    # ---- common: run ----
    logger.info(f"Build_params func will return params for {model_type}")
    params["seed"] = int(cfg.get("run").get("seed", 0))
    if model_type is None:
        raise ValueError("config missing: [run].decoder")

    # ---- common: dataset ----
    params["dataset_dir"] = cfg.get("dataset").get("dir")
    params["expt"] = cfg.get("dataset").get("expt")
    params["pattern"] = cfg.get("dataset").get("pattern", "*.mat")

    if params["dataset_dir"] is None or params["expt"] is None:
        raise ValueError("config missing: [dataset].dir and/or [dataset].expt")

    params["absolute_path"] = os.path.dirname(os.path.abspath(__file__))
    params["directory"] = os.path.join(params["absolute_path"], params["dataset_dir"], params["expt"])
    params["files"] = [
        os.path.basename(p)
        for p in sorted(glob.glob(os.path.join(params["directory"], params["pattern"])))
    ]

    # ---- parameters for model ----
    setting = {}
    setting["error"] = int(cfg.get("feedback").get("error", 0))
    setting["sparsity_rate"] = int(cfg.get("feedback").get("sparse_rate", 0))

    # ---- decoder-specific ----
    if model_type == "banditron":
        setting["gamma"] = cfg.get("banditron").get("gamma")
        setting["k"] = output_dim
        logger.info(f"Banditron - output_dim is defined as: {setting['k']}")
        if setting["gamma"] is None:
            raise ValueError("config missing: [banditron].gamma")
        setting["gamma"] = float(setting["gamma"])
    elif model_type == "banditronRP":
        setting["gamma"] = cfg.get("banditron").get("gamma")
        setting["k"] = output_dim
        logger.info(f"BanditronRP - output_dim is defined as: {setting['k']}")
        if setting["gamma"] is None:
            raise ValueError("config missing: [banditron].gamma")
        setting["gamma"] = float(setting["gamma"])

    elif model_type == "HRL":
        setting["muH"] = cfg.get("HRL").get("muH")
        setting["muO"] = cfg.get("HRL").get("muO")
        
        #to use optimized hidden layer
        width_keys = sorted([k for k in cfg.get("HRL", {}).keys() if k.startswith("width_")])
        optimized_hidden = [cfg.get("HRL").get(k) for k in width_keys]
        setting["num_nodes"] = [input_dim] + optimized_hidden + [output_dim]

        #to use hidden layer defined in toml
        # setting["num_nodes"] = cfg.get("HRL").get("num_nodes")
        # setting["num_nodes"][-1] = output_dim #change to output_dim depending on used labels
        
        logger.info(f"HRL - output_dim is defined as: {setting['num_nodes']}")

        if setting["muH"] is None or setting["muO"] is None or setting["num_nodes"] is None:
            raise ValueError("config missing: [HRL].muH, [HRL].muO, [HRL].num_nodes")

        setting["muH"] = float(setting["muH"])
        setting["muO"] = float(setting["muO"])
        setting["num_nodes"] = list(setting["num_nodes"])

    elif model_type == "AGREL":
        setting["alpha"] = float(cfg.get("AGREL").get("alpha", 0.1))
        setting["beta"] = float(cfg.get("AGREL").get("beta", 0.1))
        setting["gamma"] = float(cfg.get("AGREL").get("gamma", 0.02))

        #to use optimized hidden layer
        width_keys = sorted([k for k in cfg.get("AGREL", {}).keys() if k.startswith("width_")])
        optimized_hidden = [cfg.get("AGREL").get(k) for k in width_keys]
        setting["num_nodes"] = [input_dim] + optimized_hidden + [output_dim]

        #to use hidden layer defined in toml
        # setting["num_nodes"] = list(cfg.get("AGREL").get("num_nodes"))
        # setting["num_nodes"][-1] = output_dim #change to output_dim depending on used labels
        logger.info(f"AGREL - output_dim is defined as: {setting['num_nodes']}")

    elif model_type == "DQN":
        setting["epsilon"] = cfg.get("dqn").get("epsilon")
        setting["gamma"] = cfg.get("dqn").get("gamma")

        if setting["epsilon"] is None or setting["gamma"] is None:
            raise ValueError("config missing: [dqn].epsilon and/or [dqn].gamma")

        setting["epsilon"] = float(setting["epsilon"])
        setting["gamma"] = float(setting["gamma"])

    elif model_type == "QLGBM":
            setting["epsilon"] = cfg.get("qlgbm").get("epsilon")
            setting["gamma"] = cfg.get("qlgbm").get("gamma")

            if setting["epsilon"] is None or setting["gamma"] is None:
                raise ValueError("config missing: [qlgbm].epsilon and/or [qlgbm].gamma")

            setting["epsilon"] = float(setting["epsilon"])
            setting["gamma"] = float(setting["gamma"])

    else:
        raise ValueError(f"Unknown decoder: {params['decoder']}")
    

    # acc_DQN = analysis('e', directory, files, 1, epsilon=epsilon, gamma_DQN=gamma_DQN, error=error, sparsity_rate=sparsity_rate)
    # acc_QLGBM = analysis('f', directory, files, 1, epsilon=epsilon, gamma_DQN=gamma_DQN, error=error, sparsity_rate=sparsity_rate)
    
    params["setting"] = setting

    return params


# import tomli
# with open("/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/RLINK/src/RL_decoders/config/config.toml", "rb") as f: # Open in binary mode ('rb')
#     data = tomli.load(f)

# params = build_params(model_type="HRL", cfg=data)
# for key, value in params.items():
#     print(f"{key}: {value}")