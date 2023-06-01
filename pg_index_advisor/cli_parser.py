import argparse
import datetime

class ExplicitDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        if action.default in (None, False):
            return action.help

        help = action.help
        if '%(default)' not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += ' Default: %(default)s.'
        return help


class CLIParser:
    def __init__(self):
        pass

    def create_parser(self):
        parser = argparse.ArgumentParser(formatter_class=ExplicitDefaultsHelpFormatter)
        self._add_parser_main_arguments(parser)

        subparsers = parser.add_subparsers(help='Action help', dest='action')
        self._add_learn_action_to_parser(subparsers)
        self._add_recommend_action_to_parser(subparsers)

        return parser

    @staticmethod
    def _add_parser_main_arguments(parser):
        config_file_help = \
            'Path to the configuration file'
        parser.add_argument('-c', '--config-file',
                            help=config_file_help,
                            type=str,
                            default='/home/irina/PG-Index-Advisor/config/tpch.json',
                            dest='config')

        width_help = \
            'The maximum number of attributes in the index being created. ' \
            'The higher the value, the more time it takes to learn.'
        parser.add_argument('-w', '--width', help=width_help, type=int, default=3)

        workload_from_help = \
            'The time from which to start sampling the workload. ' \
            'The default value is the current date. ' \
            'Format: ISO 8601. ' \
            'Example: 2023-05-29T15:58:55Z.'
        parser.add_argument('-f', '--workload-from', help=workload_from_help, type=datetime.datetime.fromisoformat,
                            dest='workload_from')

        workload_to_help = \
            'The time before which the workload needs to be sampled. ' \
            'The default value is the current date. ' \
            'Format: ISO 8601. ' \
            'Example: 2023-05-29T15:58:55Z.'
        parser.add_argument('-t', '--workload-to', help=workload_to_help, type=datetime.datetime.fromisoformat,
                            dest='workload_to')

        # HYPERPARAMETERS
        hyperparameters_parser = parser.add_argument_group('hyperparameter arguments')

        random_seed_help = \
            'A random value for a random number generator.'
        hyperparameters_parser.add_argument('-r', '--random-seed', help=random_seed_help, type=int, default=0,
                                            dest='random_seed')

        workload_size_help = \
            'The number of classes in the workload. ' \
            'If no value is specified, then all classes are used.'
        hyperparameters_parser.add_argument('-q', '--query-classes', help=workload_size_help, type=int,
                                            dest='query_classes')

        workload_representation_size_help = \
            'Query text representation size.'
        hyperparameters_parser.add_argument('-R', '--representation-size', help=workload_representation_size_help,
                                            type=int, default=50, dest='representation_size')

        # DATABASE GROUP
        db_parser = parser.add_argument_group('database required arguments')

        database_help = \
            'Database name to connect to.'
        db_parser.add_argument('-d', '--database', help=database_help, type=str, required=True)

        username_help = \
            'Database username to connect to.'
        db_parser.add_argument('-u', '--username', help=username_help, type=str, required=True)

        password_help = \
            'Database password to connect to.'
        db_parser.add_argument('-p', '--password', help=password_help, type=str, required=True)

        host_help = \
            'Database host to connect to.'
        db_parser.add_argument('--host', help=host_help, type=str, default='localhost')

        port_help = \
            'Database port to connect to.'
        db_parser.add_argument('--port', help=port_help, type=int, default=5432)

        # FILTERS
        filter_parser = parser.add_argument_group('filter optional arguments')

        row_threshold_help = \
            'Filters tables in which the number of rows is less ' \
            'than the specified value.'
        filter_parser.add_argument('--row-threshold', help=row_threshold_help, type=int, default=-1,
                                   dest='row_threshold')

        allowed_table_help = \
            'Conducts training only according to the specified tables. ' \
            'For a list of several tables, the key can be ' \
            'specified several times.'
        filter_parser.add_argument('--allowed-table', help=allowed_table_help, type=str, action='append',
                                   dest='allowed_table')

    # LEARN COMMAND
    @staticmethod
    def _add_learn_action_to_parser(subparsers):
        parser_l = subparsers.add_parser('learn', help='Help for learning actions')

        max_steps_per_episode_help = \
            'The maximum number of steps per episode. ' \
            'It directly affects the duration of the episode.'
        parser_l.add_argument('-s', '--steps', help=max_steps_per_episode_help, type=int, default=1000)

        validation_frequency_help = \
            'The number of steps through which ' \
            'the performance of the model is checked.'
        parser_l.add_argument('-v', '--validation-frequency', help=validation_frequency_help, type=int, default=200,
                              dest='validation_frequency')

        timestamps_help = \
            'The number of training episodes. ' \
            'The harder the load and the more space.'
        parser_l.add_argument('-t', '--timestamps', help=timestamps_help, type=int, default=10000)

        result_path_help = \
            'The path to record the training report.'
        parser_l.add_argument('-r', '--result-path', help=result_path_help, type=str, default='', dest='result_path')

        budget_help = \
            'Budget for validation and testing. ' \
            'The key can be specified several times with different values.'
        parser_l.add_argument('-b', '--budget', help=budget_help, type=int, action='append')

    # RECOMMEND COMMAND
    @staticmethod
    def _add_recommend_action_to_parser(subparsers):
        parser_r = subparsers.add_parser('recommend', help='Help for recommendation action')
