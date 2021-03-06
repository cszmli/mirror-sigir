"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.Models import NMTModel, MeanEncoder, RNNEncoder, \
                        StdRNNDecoder, InputFeedRNNDecoder, MirrorModel, MirrorLight, Mirror_Encshare, MirrorLow_Encshare, Input_Z_FeedRNNDecoder, Input_Z_RNNDecoder, MirrorLow
from onmt.modules import Embeddings, ImageEncoder, CopyGenerator, \
                         TransformerEncoder, TransformerDecoder, \
                         CNNEncoder, CNNDecoder, AudioEncoder
from onmt.Utils import use_gpu


def make_embeddings(opt, word_dict, feature_dicts, for_encoder=True):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[onmt.io.PAD_WORD]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[onmt.io.PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]

    return Embeddings(word_vec_size=embedding_dim,
                      position_encoding=opt.position_encoding,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=opt.feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      feat_padding_idx=feats_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      feat_vocab_sizes=num_feat_embeddings)


def make_encoder(opt, embeddings,ctx=False):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    if opt.encoder_type == "transformer":
        return TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.dropout, embeddings)
    elif opt.encoder_type == "cnn":
        return CNNEncoder(opt.enc_layers, opt.rnn_size,
                          opt.cnn_kernel_width,
                          opt.dropout, embeddings)
    elif opt.encoder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings)
    else:
        # "rnn" or "brnn"
        if ctx:
            return RNNEncoder(opt.rnn_type, opt.ctx_bid, opt.enc_layers,
                          opt.rnn_size, opt.dropout, embeddings)
        else:
            return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers,
                            opt.rnn_size, opt.dropout, embeddings)


def make_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if opt.decoder_type == "transformer":
        return TransformerDecoder(opt.dec_layers, opt.rnn_size,
                                  opt.global_attention, opt.copy_attn,
                                  opt.dropout, embeddings)
    elif opt.decoder_type == "cnn":
        return CNNDecoder(opt.dec_layers, opt.rnn_size,
                          opt.global_attention, opt.copy_attn,
                          opt.cnn_kernel_width, opt.dropout,
                          embeddings)
    elif opt.input_feed:
        if opt.input_feed_with_ctx:
            return Input_Z_FeedRNNDecoder(opt.rnn_type, opt.brnn,
                                    opt.dec_layers, opt.rnn_size,
                                    opt.global_attention,
                                    opt.coverage_attn,
                                    opt.context_gate,
                                    opt.copy_attn,
                                    opt.dropout,
                                    embeddings,
                                    opt.z_dim)
        else:
            return Input_Z_RNNDecoder(opt.rnn_type, opt.brnn,
                        opt.dec_layers, opt.rnn_size,
                        opt.global_attention,
                        opt.coverage_attn,
                        opt.context_gate,
                        opt.copy_attn,
                        opt.dropout,
                        embeddings,
                        opt.z_dim)

    else:
        return StdRNNDecoder(opt.rnn_type, opt.brnn,
                             opt.dec_layers, opt.rnn_size,
                             opt.global_attention,
                             opt.coverage_attn,
                             opt.context_gate,
                             opt.copy_attn,
                             opt.dropout,
                             embeddings)


def load_test_model(opt, dummy_opt,):
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    fields = onmt.io.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    model = make_mirror_model(model_opt, fields,
                            use_gpu(opt), checkpoint, model_opt.mirror_type)

    model.eval()
    model.generator_cxz2y.eval()
    model.generator_cyz2x.eval()
    model.generator_cz2x.eval()
    model.generator_cz2y.eval()
    return fields, model, model_opt

def load_mmi_model(opt, dummy_opt):
    checkpoint = torch.load(opt.mmi_model,
                            map_location=lambda storage, loc: storage)
    fields = onmt.io.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    model = make_base_model(model_opt, fields,
                            use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt

def make_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make encoder.
    if model_opt.model_type == "text":
        src_dict = fields["src"].vocab
        feature_dicts = onmt.io.collect_feature_vocabs(fields, 'src')
        src_embeddings = make_embeddings(model_opt, src_dict,
                                         feature_dicts)
        encoder = make_encoder(model_opt, src_embeddings)
    elif model_opt.model_type == "img":
        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    # Make decoder.
    tgt_dict = fields["tgt"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, 'tgt')
    tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    decoder = make_decoder(model_opt, tgt_embeddings)

    # Make NMTModel(= encoder + decoder).
    model = NMTModel(encoder, decoder)
    model.model_type = model_opt.model_type

    # Make Generator.
    if not model_opt.copy_attn:
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(fields["tgt"].vocab)),
            nn.LogSoftmax())
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.rnn_size,
                                  fields["tgt"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model




def make_mirror_model(model_opt, fields, gpu, checkpoint=None, mirror_type='mirror'):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make encoder.
    # if model_opt.model_type == "text":
    src_dict = fields["src"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, 'src')
    src_embeddings = make_embeddings(model_opt, src_dict,
                                        feature_dicts)
    # src_embeddings_ctx = make_embeddings(model_opt, src_dict,
                                    #  feature_dicts)
    src_embeddings_ctx = src_embeddings
    encoder_utt = make_encoder(model_opt, src_embeddings)
    # encoder_utt_y = make_encoder(model_opt, src_embeddings)
    encoder_utt_y = None
    encoder_ctx = make_encoder(model_opt, src_embeddings_ctx, ctx=True)


    # Make decoder.
    tgt_dict = fields["tgt"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, 'tgt')
    tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)
    tgt_embeddings_2 = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)
    tgt_embeddings_3 = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)
    tgt_embeddings_4 = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)
    
    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        print("encoder and decoder will share the embedding")
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')
        tgt_embeddings = src_embeddings
        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight
        tgt_embeddings_2.word_lut.weight = src_embeddings.word_lut.weight
        tgt_embeddings_3.word_lut.weight = src_embeddings.word_lut.weight
        tgt_embeddings_4.word_lut.weight = src_embeddings.word_lut.weight
        src_embeddings_ctx.word_lut.weight = src_embeddings.word_lut.weight

    decoder_1 = make_decoder(model_opt, tgt_embeddings)
    decoder_2 = make_decoder(model_opt, tgt_embeddings_2)
    decoder_3 = make_decoder(model_opt, tgt_embeddings_3)
    decoder_4 = make_decoder(model_opt, tgt_embeddings_4)

    # Make NMTModel(= encoder + decoder).
    if mirror_type=='mirror':
        model = MirrorModel(encoder_utt, encoder_utt_y, encoder_ctx, decoder_1, decoder_2, decoder_3, decoder_4, opt=model_opt)
    elif mirror_type=='mirror_light':
        model = MirrorLight(encoder_utt, encoder_utt_y, encoder_ctx, decoder_1, decoder_2, decoder_3, decoder_4, opt=model_opt)
    elif mirror_type=='mirror_low':
        model = MirrorLow(encoder_utt, encoder_utt_y, encoder_ctx, decoder_1, decoder_2, decoder_3, decoder_4, opt=model_opt)
    elif mirror_type=='mirror_low_encshare':
        model = MirrorLow_Encshare(encoder_utt, decoder_1, decoder_2, decoder_3, decoder_4, opt=model_opt)
    else:
        raise ValueError("no such model type: {}".format(mirror_type))
    model.model_type = model_opt.model_type

    # Make Generator.
    generator_cxz2y = make_generator(model_opt, model.decoder_cxz2y, fields, 'tgt')
    generator_cz2x = make_generator(model_opt, model.decoder_cz2x, fields, 'tgt_back')
    generator_cyz2x = make_generator(model_opt, model.decoder_cyz2x, fields, 'tgt_back')
    generator_cz2y = make_generator(model_opt, model.decoder_cz2y, fields, 'tgt')


    # Add generator to model (this registers it as parameter of model).
    model.generator_cxz2y = generator_cxz2y
    model.generator_cz2x = generator_cz2x
    model.generator_cyz2x = generator_cyz2x
    model.generator_cz2y = generator_cz2y

    # Load the model states from checkpoint or initialize them.
    # careful: fix the loading module later
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        # generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            # for p in generator_cxz2y.parameters():
            #     p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            # for p in generator_cz2x.parameters():
            #     p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            # for p in generator_cyz2x.parameters():
            #     p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            # for p in generator_cz2y.parameters():
            #     p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.encoder_ctx, 'embeddings'):
            model.encoder_ctx.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder_cxz2y, 'embeddings'):
            model.decoder_cxz2y.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)
        if hasattr(model.decoder_cz2x, 'embeddings'):
            model.decoder_cz2x.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)
        if hasattr(model.decoder_cyz2x, 'embeddings'):
            model.decoder_cyz2x.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)
        if hasattr(model.decoder_cz2y, 'embeddings'):
            model.decoder_cz2y.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)




    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model

def make_generator(model_opt, decoder, fields, des='tgt'):
        # Make Generator.
    if not model_opt.copy_attn:
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(fields[des].vocab)),
            nn.LogSoftmax(dim=-1))
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.rnn_size,
                                  fields[des].vocab)
    return generator



