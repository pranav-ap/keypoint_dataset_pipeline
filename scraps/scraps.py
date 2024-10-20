
class DSMatcher(KeypointMatcher):
    def __init__(self):
        super().__init__()

        from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
        self.matcher = DualSoftMaxMatcher()

    """
    Match
    """

    def _match_pair(self, pair: DSM_MatchesData):
        a: KeypointsData = pair.a
        b: KeypointsData = pair.b

        matches_A, matches_B, batch_ids = self.matcher.match(
            a.keypoints, a.descriptions,
            b.keypoints, b.descriptions,
            normalize=True,
            inv_temp=config.DSMatcher_INV_TEMP,
            threshold=config.DSMatcher_THRESHOLD
        )

        H = a.image.height
        W = a.image.width

        left_matches, right_matches = self.matcher.to_pixel_coords(
            matches_A, matches_B,
            H, W, H, W
        )

        pair.set_left_matches(left_matches)
        pair.set_right_matches(right_matches)

    def extract_matches(self, image_names):
        a: Optional[KeypointsData] = None

        for index in range(len(image_names) - 1):
            name_a = image_names[index]
            name_b = image_names[index + 1]

            if a is None:
                a = KeypointsData(name_a)
                a.load()

            b = KeypointsData(name_b)
            b.load()

            pair = DSM_MatchesData(a, b)
            self._match_pair(pair)
            pair.save_matches()

            a = b

    @staticmethod
    def show_keypoint_matches(name_a, name_b, num_points=5):
        a = KeypointsData(name_a)
        a.load()

        b = KeypointsData(name_b)
        b.load()

        pair = DSM_MatchesData(a, b)
        pair.load_matches()

        return pair.plot_matches(num_points)




class DSM_MatchesData(MatchesData):
    def __init__(self, a: KeypointsData, b: KeypointsData):
        super().__init__(a, b)

        self.left_matches: Optional[torch.tensor] = None
        self.right_matches: Optional[torch.tensor] = None

    def set_left_matches(self, left_matches):
        self.left_matches = left_matches
        self.left_matches_coords = [
            cv2.KeyPoint(int(x.item()), int(y.item()), 1.)
            for x, y in self.left_matches
        ]

    def set_right_matches(self, right_matches):
        self.right_matches = right_matches
        self.right_matches_coords = [
            cv2.KeyPoint(int(x.item()), int(y.item()), 1.)
            for x, y in self.right_matches
        ]

    """
    Load & Save
    """

    def load_matches(self):
        if self.left_matches is None or self.left_matches_coords is None or self.right_matches is None or self.right_matches_coords is None:
            filename = f"{self.a.image_name}_{self.b.image_name}_matches_{config.POSTFIX_MATCHER_MODEL}_{config.POSTFIX_DATASET}.pt"
            matches = load_tensor(filename)

            self.set_left_matches(matches[:, :2])
            self.set_right_matches(matches[:, 2:])

    def save_matches(self):
        assert self.left_matches is not None
        assert self.right_matches is not None

        filename = f"{self.a.image_name}_{self.b.image_name}_matches_{config.POSTFIX_MATCHER_MODEL}_{config.POSTFIX_DATASET}.pt"
        matches = torch.cat([self.left_matches, self.right_matches], dim=1)
        save_tensor(matches, filename)



