import os, glob
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def build_params(model_type, cfg: dict) -> dict:
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
    setting["sparsity_rate"] = int(cfg.get("feedback").get("spars", 0))

    # ---- decoder-specific ----
    if model_type == "banditron":
        setting["gamma"] = cfg.get("banditron").get("gamma")
        setting["k"] = cfg.get("banditron").get("k")
        if setting["gamma"] is None:
            raise ValueError("config missing: [banditron].gamma")
        setting["gamma"] = float(setting["gamma"])

    elif model_type == "HRL":
        setting["muH"] = cfg.get("hrl").get("muH")
        setting["muO"] = cfg.get("hrl").get("muO")
        setting["num_nodes"] = cfg.get("hrl").get("num_nodes")

        if setting["muH"] is None or setting["muO"] is None or setting["num_nodes"] is None:
            raise ValueError("config missing: [hrl].muH, [hrl].muO, [hrl].num_nodes")

        setting["muH"] = float(setting["muH"])
        setting["muO"] = float(setting["muO"])
        setting["num_nodes"] = list(setting["num_nodes"])

    elif model_type == "AGREL":
        setting["alpha"] = float(cfg.get("agrel").get("alpha", 0.1))
        setting["beta"] = float(cfg.get("agrel").get("beta", 0.1))
        setting["gamma"] = float(cfg.get("agrel").get("gamma", 0.02))
        setting["num_nodes"] = list(cfg.get("agrel").get("num_nodes", [1000, 4]))

    elif model_type == "DQN":
        setting["epsilon"] = cfg.get("dqn").get("epsilon")
        setting["gamma"] = cfg.get("dqn").get("gamma")

        if setting["epsilon"] is None or setting["gamma"] is None:
            raise ValueError("config missing: [dqn].epsilon and/or [dqn].gamma")

        setting["epsilon"] = float(setting["epsilon"])
        setting["gamma"] = float(setting["gamma"])

    else:
        raise ValueError(f"Unknown decoder: {params['decoder']}")
    
    params["setting"] = setting

    return params


# import tomli
# with open("/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/RLINK/src/RL_decoders/config/config.toml", "rb") as f: # Open in binary mode ('rb')
#     data = tomli.load(f)

# params = build_params(model_type="HRL", cfg=data)
# for key, value in params.items():
#     print(f"{key}: {value}")