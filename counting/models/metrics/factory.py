from bootstrap.lib.options import Options
from .tallyqa_metrics import TallyQAMetrics


def factory(engine, mode):
    name = Options()["model.metric.name"]
    metric = None
    if name == "tallyqa_metrics":
        metric = TallyQAMetrics(mode=mode, engine=engine,)
    else:
        raise ValueError(name)
    return metric
