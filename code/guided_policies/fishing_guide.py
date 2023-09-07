class FishingGuide:
    def __init__(self, **kwargs):
        self.cooldown = 0

    def get_action(self, state, info):
        info_obs = info["obs"]
        sound_subtitles = info_obs.sound_subtitles
        bobber_thrown = info_obs.bobber_thrown
        # print(f"{bobber_thrown=}")
        self.cooldown = max(0, self.cooldown - 1)
        for sound in sound_subtitles:
            if sound.translate_key == "subtitles.entity.experience_orb.pickup":
                pass
            elif sound.translate_key == "subtitles.entity.fishing_bobber.retrieve":
                return 0
            elif sound.translate_key == "subtitles.entity.fishing_bobber.splash":
                # print("Splash, Will Retrieve")
                if bobber_thrown:
                    self.cooldown = 3
                    return 1
                else:
                    return 0
            elif sound.translate_key == "subtitles.entity.fishing_bobber.throw":
                pass
            elif sound.translate_key == "subtitles.entity.item.pickup":
                pass
        if not bobber_thrown and self.cooldown == 0:
            # print("Not Thrown, Will Throw")
            self.cooldown = 3
            return 1
        return 0

    def reset(self):
        self.cooldown = 0
