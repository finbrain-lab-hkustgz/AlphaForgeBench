from mmengine.registry import Registry

SCALER = Registry("scaler", locations=["src.dataset"])
DATASET= Registry("dataset", locations=["src.dataset"])
DOWNLOADER = Registry("downloader", locations=["src.download"])
PROCESSOR = Registry("processor", locations=["src.process"])

TOOL = Registry("tool", locations=["src.tool"])
ENVIRONMENT = Registry("environment", locations=["src.environment"])
METRIC = Registry("metric", locations=["src.metric"])