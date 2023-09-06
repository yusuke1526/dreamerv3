# 1. Test setup:
# docker run -it --rm --gpus all nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04 nvidia-smi
#
# If the above does not work, try adding the --privileged flag
# and changing the command to `sh -c 'ldconfig -v && nvidia-smi'`.
#
# 2. Start training:
# docker build -f  dreamerv3/Dockerfile -t img . && \
# docker run -it --rm --gpus all -v ~/logdir:/logdir img \
#   sh /scripts/xvfb_run.sh python3 dreamerv3/train.py \
#   --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
#   --configs dmc_vision --task dmc_walker_walk
#
# 3. See results:
# tensorboard --logdir ~/logdir

# System
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
RUN apt-get update -y && apt-get install -y freeglut3-dev mesa-utils wget
RUN wget --content-disposition https://sourceforge.net/projects/virtualgl/files/2.6.5/virtualgl_2.6.5_amd64.deb/download
RUN apt-get install -y libxtst6 libxv1 xauth x11-xkb-utils xkb-data
RUN dpkg -r virtualgl && dpkg -i virtualgl_2.6.5_amd64.deb
RUN wget --content-disposition https://sourceforge.net/projects/turbovnc/files/2.2.7/turbovnc_2.2.7_amd64.deb/download
RUN dpkg -i turbovnc_2.2.7_amd64.deb
RUN apt-get install -y x11-xserver-utils libxrandr-dev
RUN apt-get remove nvidia* && apt-get install -y nvidia-driver-430

# Display
RUN nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
RUN nohup Xorg :7 &
RUN echo "password\npassword\n" | /opt/TurboVNC/bin/vncserver :8 &
RUN /opt/TurboVNC/bin/vncserver -list
RUN DISPLAY=:8 vglrun -d :7.0 glxinfo

CMD DISPLAY=:8 vglrun -d :7.0 /CARLA/CarlaUE4.sh