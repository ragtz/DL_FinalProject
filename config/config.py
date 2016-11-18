from DL_FinalProject.config import test_config
from DL_FinalProject.config import model1
from DL_FinalProject.config import model2
from DL_FinalProject.config import model3
from DL_FinalProject.config import model4
from DL_FinalProject.config import model9
from DL_FinalProject.config import lstmgan_test

configs = {'test': test_config.TestLSTMConfig, 
           'model1': model1.ModelOneLSTMConfig,
           'model2': model2.ModelTwoLSTMConfig,
           'model3': model3.ModelThreeLSTMConfig,
           'model4': model4.ModelFourLSTMConfig,
           'model9': model9.ModelNineLSTMConfig,
           'lstmgan_test': lstmgan_test.LSTMGANTestConfig}

