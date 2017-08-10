import os
import argparse
import shutil

from cli.mmt import BilingualCorpus
from cli.mmt.neural import OpenNMTDecoder, OpenNMTPreprocessor

import torch

class OpenNMTEngine:
    #def __init__(self, source_lang, target_lang, decoder_path):
    def __init__(self, opt):


#args.source_lang, args.target_lang, args.model_path


        # Neural specific processing (BPE)
        self.onmt_preprocessor = OpenNMTPreprocessor(opt.source_lang, opt.target_lang, os.path.join(opt.model_path, 'model.bpe'))

        # Neural engine
        self.decoder = OpenNMTDecoder(os.path.join(opt.model_path, 'model.pt'), opt.source_lang, opt.target_lang, opt.gpus)

    #     def _build_memory(self, args, skip=False, log=None):
    #         if not skip:
    #             corpora = filter(None, [args.filtered_bilingual_corpora, args.processed_bilingual_corpora,
    #                                     args.bilingual_corpora])[0]
    #
    #             self._engine.memory.create(corpora, log=log)

    def _prepare_training_data(self, opt, skip=False):

        training_corpora = BilingualCorpus.list(opt.training_corpora)
        print 'training_corpora:%s' % repr(training_corpora)
        valid_corpora = BilingualCorpus.list(opt.valid_corpora)
        print 'valid_corpora:%s' % repr(valid_corpora)

        if not skip:
            self.onmt_preprocessor.process(training_corpora, valid_corpora, opt.training_dir,
                                           bpe_symbols=opt.bpe_symbols, max_vocab_size=opt.bpe_max_vocab_size,
                                           working_dir=opt.training_dir)

    def _train_decoder(self, opt, skip=False, delete_on_exit=False):

        if not skip:
            self.decoder.train(opt.training_dir, opt.training_dir)

            if delete_on_exit:
                shutil.rmtree(opt.training_dir, ignore_errors=True)



def run_main():
    # Args parse
    # ------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='train a neural network')

    parser.add_argument('-source_lang', required=True,
                    help="Source language")
    parser.add_argument('-target_lang', required=True,
                    help="Target language")
    parser.add_argument('-training_corpora', required=True,
                    help="Path to the training data")
    parser.add_argument('-valid_corpora', required=True,
                    help="Path to the validation data")
    parser.add_argument('-model_path', required=True,
                    help="Path to the models")
    parser.add_argument('-training_dir', required=True,
                    help="Path to the models")

    parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
    
    parser.add_argument('-l', '--log-level', dest='log_level', metavar='LEVEL', help='select the log level',
                        choices=['critical', 'error', 'warning', 'info', 'debug'], default='info')

    args = parser.parse_args()
    args.bpe_symbols = 32000
    args.bpe_max_vocab_size = 90000


    if torch.cuda.is_available():
        if args.gpus is None:
            gpus = range(torch.cuda.device_count()) if torch.cuda.is_available() else None
        else:
            # remove indexes of GPUs which are not valid,
            # because larger than the number of available GPU or smaller than 0
            gpus = [x for x in args.gpus if x < torch.cuda.device_count() or x < 0]
            if len(gpus) == 0:
                gpus = None
    else:
        gpus = None

    decoder_path = os.path.join(args.model_path, 'decoder')

    # retrival engine to get suggestion for instance-based adaptation
    # memory = TranslationMemory(os.path.join(decoder_path, 'memory'), self.source_lang, self.target_lang)

    engine = OpenNMTEngine(args)

    engine._prepare_training_data(args, skip=False)

    engine._train_decoder(args, skip=False)


if __name__ == '__main__':
    run_main()