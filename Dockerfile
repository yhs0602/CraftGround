# Ubuntu 베이스 이미지 선택
FROM ubuntu:latest

# 환경 설정 및 필요한 소프트웨어 설치
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk python3-pip git && \
    apt-get clean

# pip를 사용하여 GitHub에서 Python 패키지 직접 설치
RUN pip3 install git+https://github.com/yhs0602/CraftGround

# Python 사이트 패키지 경로를 환경변수에 추가 (pip 패키지 경로를 찾기 위함)
ENV PYTHON_SITE_PACKAGES=/usr/local/lib/python3.*/dist-packages

# craftground/MinecraftEnv 디렉토리로 이동하여 gradlew build 실행
RUN cd $(find $PYTHON_SITE_PACKAGES -type d -maxdepth 1 -name "craftground")/MinecraftEnv && chmod +x gradlew && ./gradlew build

# 실험 파일이 추가될 작업 디렉토리 설정
WORKDIR /workspace

# 컨테이너 실행 시 기본 명령 설정
CMD ["bash"]
