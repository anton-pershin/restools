import unittest
import subprocess

class LauncherOfESNEnsembleGoodCheck(unittest.TestCase):
    
    def test_launcher(self):
        command_line = 'python papers/none2021_predicting_transition_using_reservoir_computing/launchers/launcher_of_ESN_ensemble.py'
        res = subprocess.run(command_line, shell=True)
        return

#if __name__ == '__main__':
#    unittest.main()
