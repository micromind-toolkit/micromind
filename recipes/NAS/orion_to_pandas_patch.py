from orion.core.worker.experiment import Experiment
import pandas
from orion.core.utils.flatten import flatten


class orion_patch(Experiment):
    def __init__(self, experiment):
        # Extract the required attributes from the experiment object
        attributes = [
            "name",
            "space",
            "version",
            "mode",
            "_id",
            "max_trials",
            "max_broken",
            "algorithm",
            "working_dir",
            "metadata",
            "refers",
            "storage",
        ]
        kwargs = {attr: getattr(experiment, attr) for attr in attributes}

        # Call the parent class constructor with the extracted attributes
        super().__init__(**kwargs)

    def to_pandas(self, with_evc_tree=False):
        """Builds a dataframe with the trials of the experiment

        Parameters
        ----------
        with_evc_tree: bool, optional
            Fetch all trials from the EVC tree.
            Default: False

        """
        columns = [
            "id",
            "experiment_id",
            "status",
            "suggested",
            "reserved",
            "completed",
            "objective",
        ]

        data = []
        for trial in self.fetch_trials(with_evc_tree=with_evc_tree):
            row = [
                trial.id,
                trial.experiment,
                trial.status,
                trial.submit_time,
                trial.start_time,
                trial.end_time,
            ]
            row.append(trial.objective.value if trial.objective else None)
            params = flatten(trial.params)
            for name in self.space.keys():
                row.append(params[name])
            for stats in trial.statistics:
                row.append(stats.value)
            data.append(row)

        columns += list(self.space.keys())
        columns += list(stats.name for stats in trial.statistics)

        if not data:
            return pandas.DataFrame([], columns=columns)

        return pandas.DataFrame(data, columns=columns)
