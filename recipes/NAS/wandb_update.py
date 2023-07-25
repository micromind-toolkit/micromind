import wandb

api = wandb.Api()
run = api.run("mariam-jamal/cifar10/pyorfvoi")
run.summary.update({"Params": 0.244375})
