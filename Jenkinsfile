def branchName = "${env.BRANCH_NAME}"
def packagePath = "nlu_metrics"
def VENV = ". venv/bin/activate"


def version(path) {
    readFile("${path}/__version__").split("\n")[0]
}


node('tobor1') {
    stage('Checkout') {
        deleteDir()
        checkout scm
    }

    stage('Tests') {
        sh """
    	virtualenv venv
    	${VENV}
    	pip install tox
    	tox
    	"""
    }

    stage('Publish') {
        switch (branchName) {
            case "master":
                deleteDir()
                checkout scm
                def rootPath = pwd()
                def path = "${rootPath}/${packagePath}"
                def credentials = "${env.NEXUS_USERNAME_PYPI}:${env.NEXUS_PASSWORD_PYPI}"
                sh """
                virtualenv venv
                ${VENV}
                python setup.py bdist_wheel --universal upload -r pypisnips
                git tag ${version(path)}
                git remote rm origin
                git remote add origin 'git@github.com:snipsco/nlu-metrics.git'
                git config --global user.email 'jenkins@snips.ai'
                git config --global user.name 'Jenkins'
                git push --tags
                """
            default:
                sh """
                ${VENV}
                python setup.py bdist_wheel
                """
        }
    }
}
