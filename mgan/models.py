from fairseq.models.lstm \
        import LSTMEncoder, \
               LSTMDecoder, \
               LSTMModel

from fairseq.models.fairseq_model \
        import FairseqModel


class MGANEncoder(LSTMEncoder): pass
class MGANDecoder(LSTMDecoder): pass
class MaskedMLE(LSTMModel): pass

