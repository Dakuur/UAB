#################### Configuració de RAID ####################

# Iniciar sessió com a root
sudo -i

# Crear particions en cada disc (exemple per a sdx, repetir x3)
fdisk /dev/sdx
# Dins de fdisk, executa les següents opcions:
# => n (nova partició)
# => p (partició primària)
# => t (canviar tipus de partició)
# => L (llistar tipus)
# => fd (seleccionar Linux RAID auto)
# => w (escriure i sortir)

# Formatar cada partició creada
mkfs.ext4 /dev/sdb1
mkfs.ext4 /dev/sdc1
mkfs.ext4 /dev/sdd1

# Configurar RAID 5 amb els discs
mdadm --create /dev/md0 --level=5 --raid-devices=3 /dev/sdb1 /dev/sdc1 /dev/sdd1

# Verificar l'estat de RAID
cat /proc/mdstat

# Formatejar disc raid amb ext4
sudo mkfs.ext4 /dev/md0

# Crear punt de muntatge per a RAID
mkdir /mnt/md0

# Montar RAID
mount /dev/md0 /mnt/md0

# Verificar que el RAID es munta correctament
df -h

# Automatitzar munatge de RAID
blkid /dev/md127
echo "UUID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx /mnt/md0 ext4 defaults 0 0" >> /etc/fstab





#################### Configuració de GlusterFS a MVA i MVB ####################

# A MVA:

# Instal·lar GlusterFS
apt update
apt install -y glusterfs-server glusterfs-client

# Crear i formatar particions en cada disc
fdisk /dev/sda
fdisk /dev/sdb
fdisk /dev/sdc
mkfs.ext4 /dev/sda1
mkfs.ext4 /dev/sdb1
mkfs.ext4 /dev/sdc1

# Crear directoris de muntatge
mkdir -p /data/node1 /data/node2 /data/node3

# Afegir els punts de muntatge a /etc/fstab
echo "/dev/sda1 /data/node1 ext4 defaults 0 1" >> /etc/fstab
echo "/dev/sdb1 /data/node2 ext4 defaults 0 1" >> /etc/fstab
echo "/dev/sdc1 /data/node3 ext4 defaults 0 1" >> /etc/fstab
mount -a

# Iniciar i habilitar GlusterFS
systemctl start glusterd
systemctl enable glusterd

# Afegir MVB com a node en el clúster de GlusterFS
gluster peer probe mvb




# A MVB:

apt update
apt install -y glusterfs-server glusterfs-client

# Crear particions i directoris de muntatge similars a MVA
fdisk /dev/sda
fdisk /dev/sdb
fdisk /dev/sdc
mkfs.ext4 /dev/sda1
mkfs.ext4 /dev/sdb1
mkfs.ext4 /dev/sdc1
mkdir -p /data/node4 /data/node5 /data/node6

# Afegir punts de muntatge a /etc/fstab
echo "/dev/sda1 /data/node4 ext4 defaults 0 1" >> /etc/fstab
echo "/dev/sdb1 /data/node5 ext4 defaults 0 1" >> /etc/fstab
echo "/dev/sdc1 /data/node6 ext4 defaults 0 1" >> /etc/fstab
mount -a

# Establir connexió amb MVA
gluster peer probe mva

# Crear i muntar el volum replicat
gluster volume create tv1 replica 2 mva:/data/node1/brick0 mvb:/data/node4/brick0
gluster volume start tv1






#################### Configuració de LVM ####################

# Crear volums físics en cada disc
pvcreate /dev/sda1 /dev/sdb1

# Crear grup de volums
vgcreate vg_data /dev/sda1 /dev/sdb1

# Crear volum lògic inicial amb 2 discs
lvcreate -L 30M -n lv_data vg_data

# Expansió per afegir un tercer disc
pvcreate /dev/sdc1
vgextend vg_data /dev/sdc1
lvextend -L +20M /dev/vg_data/lv_data






#################### Proves de rendiment ####################

# Crear fitxer de prova de 42 MB per a l'anàlisi
truncate --size 42M sample.txt

# Mesurar rendiment copiant el fitxer a RAID, GlusterFS i LVM
time cp sample.txt /mnt/md0   # Per a RAID
time cp sample.txt /data-client # Per a GlusterFS
time cp sample.txt /mnt/lv_data # Per a LVM
