class FishingGuide:
    def __init__(self, **kwargs):
        self.is_rod_thrown = False

    def get_action(self, state, info):
        info_obs = info["obs"]
        sound_subtitles = info_obs.sound_subtitles
        for sound in sound_subtitles:
            if sound.translate_key == "subtitles.entity.experience_orb.pickup":
                pass
            elif sound.translate_key == "subtitles.entity.fishing_bobber.retrieve":
                return 0
            elif sound.translate_key == "subtitles.entity.fishing_bobber.splash":
                # print("Splash, Will Retrieve")
                if self.is_rod_thrown:
                    self.is_rod_thrown = False
                    return 1
                else:
                    return 0
            elif sound.translate_key == "subtitles.entity.fishing_bobber.throw":
                pass
            elif sound.translate_key == "subtitles.entity.item.pickup":
                pass
        if not self.is_rod_thrown:
            self.is_rod_thrown = True
            return 1
        return 0

    def reset(self):
        self.is_rod_thrown = False
