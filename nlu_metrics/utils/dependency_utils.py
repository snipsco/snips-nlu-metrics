from subprocess import check_output, CalledProcessError

import pip


def update_nlu_packages(snips_nlu_version, snips_nlu_rust_version):
    update_package(package_name='snips_nlu', version=snips_nlu_version)
    update_package(package_name='snips_nlu_rust',
                   version=snips_nlu_rust_version)


def update_package(package_name, version):
    installed_packages = pip.get_installed_distributions()
    if any(package.key == package_name and package.version == version
           for package in installed_packages):
        return
    if any(package.key == package_name for package in installed_packages):
        uninstall_package(package_name)
    install_package(package_name, version)


def uninstall_package(package_name):
    try:
        check_output(['pip', 'uninstall', package_name, '-y'])
    except CalledProcessError as e:
        raise Exception("Error while uninstalling '%s': %s"
                        % (package_name, e.output))


def install_package(package_name, version):
    try:
        check_output(['pip', 'install', package_name + "==" + version])
    except CalledProcessError as e:
        raise Exception('Error while installing %s==%s package: %s'
                        % (package_name, version, e.output))
