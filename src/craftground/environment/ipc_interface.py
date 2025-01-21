from abc import ABC, abstractmethod
import os
import subprocess
from time import sleep

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

    @abstractmethod
    def is_alive(self) -> bool:
        pass

    @abstractmethod
    def destroy(self):
        pass

    def ensure_alive(self, fast_reset, extra_commands, seed):
        if not self.is_alive():  # first time
            self.start_server(
                port=self.port,
                use_vglrun=self.use_vglrun,
                track_native_memory=self.track_native_memory,
                ld_preload=self.ld_preload,
                seed=seed,
            )
        elif not fast_reset:
            self.destroy()
            self.start_server(
                port=self.port,
                use_vglrun=self.use_vglrun,
                track_native_memory=self.track_native_memory,
                ld_preload=self.ld_preload,
            )
        else:
            with self.csv_logger.profile("fast_reset"):
                self.send_fastreset2(self.sock, extra_commands)
            self.csv_logger.log("Sent fast reset")

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
