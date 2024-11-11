import random

import h5py

from config import config


class DataStore:
    def __init__(self, mode='a'):
        assert mode == 'r' or mode == 'a'
        self.mode = mode

    def init(self):
        filename_inter = 'data.hdf5'
        filepath_inter = f'{config.paths[config.task.name].output}/{filename_inter}'
        self._file = h5py.File(filepath_inter, self.mode)

        if self.mode == 'r':
            self._init_groups_read_mode()
        else:
            self._init_groups_append_mode()

    def _init_groups_append_mode(self):
        self._detector = self._file.create_group(f'{config.task.cam}/detector')
        self._matcher = self._file.create_group(f'{config.task.cam}/matcher')
        self._filter = self._file.create_group(f'{config.task.cam}/filter')

        # Create groups in the results file
        self._results_matches = self._file.create_group(f'{config.task.cam}/matches')

        # Setup 'detector' subgroups
        self.detector_image_level_normalised = self._detector.create_group('image_level/normalised')
        self.detector_image_level_confidences = self._detector.create_group('image_level/confidences')
        self.detector_patch_level_normalised = self._detector.create_group('patch_level/normalised')
        self.detector_patch_level_confidences = self._detector.create_group('patch_level/confidences')
        self.detector_patch_level_which_patch = self._detector.create_group('patch_level/which_patch')

        # Setup 'matcher' subgroups
        self.matcher_warp = self._matcher.create_group('warp')
        self.matcher_certainty = self._matcher.create_group('certainty')

        # Setup 'filter' subgroups
        self.filter_image_level_normalised = self._filter.create_group('image_level/normalised')
        self.filter_image_level_confidences = self._filter.create_group('image_level/confidences')
        self.filter_patch_level_normalised = self._filter.create_group('patch_level/normalised')
        self.filter_patch_level_confidences = self._filter.create_group('patch_level/confidences')
        self.filter_patch_level_which_patch = self._filter.create_group('patch_level/which_patch')

        # Setup results subgroups
        self.results_original_reference_coords = self._results_matches.create_group('original/reference_coords')
        self.results_original_target_coords = self._results_matches.create_group('original/target_coords')
        self.results_small_reference_coords = self._results_matches.create_group('small/reference_coords')
        self.results_small_target_coords = self._results_matches.create_group('small/target_coords')

    def _init_groups_read_mode(self):
        self._detector = self._file[f'{config.task.cam}/detector']
        self._matcher = self._file[f'{config.task.cam}/matcher']
        self._filter = self._file[f'{config.task.cam}/filter']

        # Create groups in the results file
        self._results_matches = self._file[f'{config.task.cam}/matches']

        # Setup 'detector' subgroups
        self.detector_image_level_normalised = self._detector['image_level/normalised']
        self.detector_image_level_confidences = self._detector['image_level/confidences']
        self.detector_patch_level_normalised = self._detector['patch_level/normalised']
        self.detector_patch_level_confidences = self._detector['patch_level/confidences']
        self.detector_patch_level_which_patch = self._detector['patch_level/which_patch']

        # Setup 'matcher' subgroups
        self.matcher_warp = self._matcher['warp']
        self.matcher_certainty = self._matcher['certainty']

        # Setup 'filter' subgroups
        self.filter_image_level_normalised = self._filter['image_level/normalised']
        self.filter_image_level_confidences = self._filter['image_level/confidences']
        self.filter_patch_level_normalised = self._filter['patch_level/normalised']
        self.filter_patch_level_confidences = self._filter['patch_level/confidences']
        self.filter_patch_level_which_patch = self._filter['patch_level/which_patch']

        # Setup results subgroups
        self.results_original_reference_coords = self._results_matches['original/reference_coords']
        self.results_original_target_coords = self._results_matches['original/target_coords']
        self.results_small_reference_coords = self._results_matches['small/reference_coords']
        self.results_small_target_coords = self._results_matches['small/target_coords']

    def get_random_pair(self):
        keys = list(self.results_small_reference_coords.keys())
        random_key = random.choice(keys)
        return random_key

    def close(self):
        self._file.close()
