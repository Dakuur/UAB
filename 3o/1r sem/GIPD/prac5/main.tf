# Definición de la imagen (Image) que se utilizará para las VMs
data "opennebula_image" "ubuntu_image" {
  name = "Ubu24.04v1.3"  # Nombre de la imagen en OpenNebula
}

# Definición de la plantilla (Template) que se utilizará para las VMs
data "opennebula_template" "ubuntu_template" {
  name = "Ubuntu 24.04-GIxPD"  # Nombre de la plantilla en OpenNebula
}

# Obtener los datos de la xarxa (red) "Internet"
data "opennebula_virtual_network" "internet_network" {
  name = "Internet"
}

# Obtener los datos del grup de seguretat (grupo de seguridad) "Default"
data "opennebula_security_group" "default_security_group" {
  name = "default"
}

resource "opennebula_virtual_machine" "vm" {
  count = 2
  name  = "virtual-machine-${count.index}"
  description = "Màquina Virtual creada amb Terraform"
  cpu = 0.5
  vcpu = 2
  memory = 1024
  permissions = "660"
  disk {
    image_id = data.opennebula_image.ubuntu_image.id
    size = 20000
    target = "vda"
    driver = "qcow2"
  }

  graphics {
    type = "VNC"
    listen = "0.0.0.0"
    keymap = "es"
  }

  # Configurar la interfície de xarxa
  nic {
    network_id = data.opennebula_virtual_network.internet_network.id # Utiliza el ID de la xarxa Internet
    model = "virtio" # Utiliza el model "virtio"
    security_groups = [data.opennebula_security_group.default_security_group.id] # Utilitza el grup de seguretat "Default"
  }

  os {
    arch = "x86_64"
    boot = "disk0"
  }

  template_id = data.opennebula_template.ubuntu_template.id

  provisioner "local-exec" {
      command = "ssh-keyscan ${self.nic[0].computed_ip} >> /home/adminp/.ssh/known_hosts"
  }

}

# Generar el archivo de inventario Ansible
resource "local_file" "ansible_inventory" {
  content = templatefile("${path.module}/templates/hosts.tpl", {
    vms = opennebula_virtual_machine.vm
  })
  filename = "${path.module}/hosts"
}

# Mostrar la dirección IP de cada VM
output "vm_ips" {
  value = [for vm in opennebula_virtual_machine.vm : vm.nic[0].computed_ip]
}