import gym


class RenderObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        return self.render_obs(observation)

    def render_obs(self, observation):
        assert self.viewer.cam.azimuth == 90
        assert self.viewer.cam.elevation == -60
        assert self.viewer.cam.distance == 6
        self.viewer.cam.lookat[0] = observation[0]
        self.viewer.cam.lookat[1] = observation[1]
        self.viewer.cam.lookat[2] = 0
        return {
            "pixels": self.render(mode="rgb_array", width=64, height=64)[
                ..., None
            ],  # last dim for stacking
            "state": observation[2:],
            "position": observation[:2],
        }
