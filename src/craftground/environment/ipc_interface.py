from abc import ABC, abstractmethod
import os
import subprocess

from proto.action_space_pb2 import ActionSpaceMessageV2
from proto.initial_environment_pb2 import InitialEnvironmentMessage
from proto.observation_space_pb2 import ObservationSpaceMessage


class IPCInterface(ABC):
    port: int

    @abstractmethod
    def send_action(self, message: ActionSpaceMessageV2):
        pass

    @abstractmethod
    def read_observation(self) -> ObservationSpaceMessage:
        pass

    @abstractmethod
    def send_initial_environment(self, message: InitialEnvironmentMessage):
        pass

    def start_server(
        self,
        env_path,
        track_native_memory: bool,
        verbose_jvm: bool,
        use_vglrun,
        ld_preload,
        options_txt_path,
        verbose_gradle: bool = False,
    ):
        my_env = os.environ.copy()
        my_env["PORT"] = str(self.port)
        my_env["VERBOSE"] = str(int(verbose_jvm))
        if track_native_memory:
            my_env["CRAFTGROUND_JVM_NATIVE_TRACKING"] = "detail"
        if self.native_debug:
            my_env["CRAFGROUND_NATIVE_DEBUG"] = "True"
        # configure permission of the gradlew
        gradlew_path = os.path.join(env_path, "gradlew")
        if not os.access(gradlew_path, os.X_OK):
            os.chmod(gradlew_path, 0o755)
        # update image settings of options.txt if exists
        if options_txt_path is not None:
            if os.path.exists(options_txt_path):
                pass
                # self.update_override_resolutions(options_txt_path)

        cmd = f"./gradlew runClient -w --no-daemon"  #  --args="--width {self.initial_env.imageSizeX} --height {self.initial_env.imageSizeY}"'
        if use_vglrun:
            cmd = f"vglrun {cmd}"
        if ld_preload:
            my_env["LD_PRELOAD"] = ld_preload
        print(f"{cmd=}")
        self.process = subprocess.Popen(
            cmd,
            cwd=self.env_path,
            shell=True,
            stdout=subprocess.DEVNULL if not verbose_gradle else None,
            env=my_env,
        )
