class FishingGuide:
    def __init__(self, **kwargs):
        pass

    def get_action(self, state, info):
        info_obs = info["obs"]
        sound_subtitles = info_obs.sound_subtitles
        for sound in sound_subtitles:
            print(sound)
            if sound.translate_key == "subtitles.entity.experience_orb.pickup":
                pass
            elif sound.translate_key == "subtitles.entity.fishing_bobber.retrieve":
                return 1
            elif sound.translate_key == "subtitles.entity.fishing_bobber.splash":
                return 1
            elif sound.translate_key == "subtitles.entity.fishing_bobber.throw":
                pass
            elif sound.translate_key == "subtitles.entity.item.pickup":
                pass

        return 0
