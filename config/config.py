from DL_FinalProject.config import test_config
from DL_FinalProject.config import model1
from DL_FinalProject.config import model2
from DL_FinalProject.config import model3
from DL_FinalProject.config import model4
from DL_FinalProject.config import model9
from DL_FinalProject.config import lstmgan_test
from DL_FinalProject.config import gan11
from DL_FinalProject.config import gan19
from DL_FinalProject.config import gan55
from DL_FinalProject.config import gan91
from DL_FinalProject.config import gan1001
from DL_FinalProject.config import gan1999
from DL_FinalProject.config import gan9991
from DL_FinalProject.config import gan2575
from DL_FinalProject.config import gan7525
from DL_FinalProject.config import ganfd
from DL_FinalProject.config import gansdg

configs = {'test': test_config.TestLSTMConfig, 
           'model1': model1.ModelOneLSTMConfig,
           'model2': model2.ModelTwoLSTMConfig,
           'model3': model3.ModelThreeLSTMConfig,
           'model4': model4.ModelFourLSTMConfig,
           'model9': model9.ModelNineLSTMConfig,
           'lstmgan_test': lstmgan_test.LSTMGANTestConfig,
           'gan11': gan11.LSTMGAN11Config,
           'gan19': gan19.LSTMGAN19Config,
           'gan55': gan55.LSTMGAN55Config,
           'gan91': gan91.LSTMGAN91Config,
           'gan1001': gan1001.LSTMGAN1001Config,
           'gan1999': gan1999.LSTMGAN1999Config,
           'gan9991': gan9991.LSTMGAN9991Config,
           'gan2575': gan2575.LSTMGAN2575Config,
           'gan7525': gan7525.LSTMGAN7525Config,
           'ganfd': ganfd.LSTMGANFDConfig,
           'gansdg': gansdg.LSTMGANSDGConfig}

