
umount /dev/sdb1
mkfs.vfat -F16 -n BOOT /dev/sdb1
mkdir -p /media/boot
mount /dev/sdb1 /media/boot
cd /media
gunzip -cd /home/liurenju/Desktop/research/mobisys-2020/rp3-optee/build/../out-br/images/rootfs.cpio.gz | sudo cpio -idmv "boot/*"
umount boot



umount /dev/sdb2
mkfs.ext4 -L rootfs /dev/sdb2


mkdir -p /media/rootfs
mount /dev/sdb2 /media/rootfs
cd rootfs
gunzip -cd /home/liurenju/Desktop/research/mobisys-2020/rp3-optee/build/../out-br/images/rootfs.cpio.gz | sudo cpio -idmv
rm -rf /media/rootfs/boot/*
cd .. && umount rootfs


# To enable Ethernet. Update the ip address if necessary.
ifconfig eth0 172.17.52.7 up

#checking
ifconfig
ping 172.17.52.6 # my desktop


# Compile ARM Compute Library
scons arch=arm64-v8a neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" -j8 internal_only=0


# Copy the binaries to the board with SD card.
cd /home/liurenju/Desktop/research/mobisys-2020/ML-examples/armnn-mnist
cp *.so* /media/liurenju/CNM/
cp mnist_caffe /media/liurenju/CNM/
cp -rf model /media/liurenju/CNM/
cp -rf data /media/liurenju/CNM/

umount /dev/sdb1




mkdir /media/ff
mount /dev/mmcblk0p1 /media/ff
cp /media/ff/* /root -rf
cd
export LD_LIBRARY_PATH=`pwd`

ln -s libmali.so ./libGLESv2.so.2
ln -s libGLESv2.so.2 ./libGLESv2.so
ln -s libmali.so ./libGLESv1_CM.so.1 
ln -s libGLESv1_CM.so.1 ./libGLESv1_CM.so
ln -s libmali.so ./libEGL.so.1
ln -s libEGL.so.1 ./libEGL.so
ln -s libmali.so ./libOpenCL.so.1
ln -s libOpenCL.so.1 ./libOpenCL.so
