[all:vars]
ansible_connection=ssh
ansible_user=adminp
ansible_ssh_pass=NebulaCaos
ansible_sudo_pass=NebulaCaos

[all]
%{ for vm in vms ~}
${vm.nic[0].computed_ip}
%{ endfor ~}