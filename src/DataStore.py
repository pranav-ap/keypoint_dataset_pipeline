import random
import h5py
from config import config


class DataStore:
    def __init__(self, mode='a'):
        assert mode == 'r' or mode == 'a'
        self.mode = mode

    def init(self, for_all_missing=False):
        filename = 'data.hdf5'
        filepath = f'{config.paths[config.task.name].output}/{filename}'
        # filepath = filepath.replace('_test', '')

        # try:
        #     with h5py.File(filepath, self.mode) as f:
        #         self._file = f 
        #         print(f.keys()) 
        # except Exception as e:
        #     print("Error opening HDF5 file:", e)
        #     exit(1)

        # noinspection PyAttributeOutsideInit
        self._file = h5py.File(filepath, self.mode)

        if for_all_missing:
            if self.mode == 'r':
                self._init_groups_all_missing_read_mode()
            else:
                self._init_groups_all_missing_append_mode()
            return

        if self.mode == 'r':
            self._init_groups_read_mode()
        else:
            # self._init_groups_read_mode()
            self._init_groups_append_mode()

    def _init_groups_all_missing_append_mode(self):
        self._matcher = self._file.create_group(f'{config.task.cam}/matcher')

        self.matcher_warp = self._matcher.create_group('warp')
        self.matcher_certainty = self._matcher.create_group('certainty')
        self.matcher_saves = self._matcher.create_group('saves')

    def _init_groups_all_missing_read_mode(self):
        self._matcher = self._file[f'{config.task.cam}/matcher']

        self.matcher_warp = self._matcher['warp']
        self.matcher_certainty = self._matcher['certainty']
        self.matcher_saves = self._matcher['saves']
        
    def _init_groups_append_mode(self):
        self._detector = self._file.create_group(f'{config.task.cam}/detector')
        self._matcher = self._file.create_group(f'{config.task.cam}/matcher')
        self._filter = self._file.create_group(f'{config.task.cam}/filter')
        self._matches = self._file.create_group(f'{config.task.cam}/matches')
        self.rotations = self._file.create_group(f'{config.task.cam}/rotationssss')

        self.detector_normalised = self._detector.create_group('normalised')
        self.detector_confidences = self._detector.create_group('confidences')

        self.matcher_warp = self._matcher.create_group('warp')
        self.matcher_certainty = self._matcher.create_group('certainty')

        self.filter_normalised = self._filter.create_group('normalised')
        self.filter_confidences = self._filter.create_group('confidences')

        self.crop_reference_coords = self._matches.create_group('crop/reference_coords')
        self.crop_target_coords = self._matches.create_group('crop/target_coords')

    def _init_groups_read_mode(self):
        self._detector = self._file[f'{config.task.cam}/detector']
        self._matcher = self._file[f'{config.task.cam}/matcher']
        self._filter = self._file[f'{config.task.cam}/filter']
        self._matches = self._file[f'{config.task.cam}/matches']
        self.rotations = self._file[f'{config.task.cam}/rotationssss']

        self.detector_normalised = self._detector['normalised']
        self.detector_confidences = self._detector['confidences']

        self.matcher_warp = self._matcher['warp']
        self.matcher_certainty = self._matcher['certainty']

        self.filter_normalised = self._filter['normalised']
        self.filter_confidences = self._filter['confidences']

        self.crop_reference_coords = self._matches['crop/reference_coords']
        self.crop_target_coords = self._matches['crop/target_coords']

    def get_random_pair(self):
        keys = list(self.crop_reference_coords.keys())
        random_key = random.choice(keys)
        return random_key

    def close(self):
        self._file.close()
