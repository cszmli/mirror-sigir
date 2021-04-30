from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import logging
import onmt
from onmt.Utils import aeq

class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack

class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :obj:`onmt.Models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Context]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """
    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (:obj:`LongTensor`):
               padded sequences of sparse indices `[src_len x batch x nfeat]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`
            hidden (class specific):
               initial hidden state.

        Returns:k
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                   `[layers x batch x hidden]`
                * contexts for attention, `[src_len x batch x hidden]`
        """
        raise NotImplementedError


class MeanEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, num_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings

    def forward(self, input, lengths=None, hidden=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()
        mean = emb.mean(0).expand(self.num_layers, batch, emb_dim)
        return (mean, mean), emb


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.no_pack_padded_seq = False

        # Use pytorch version when available.
        if rnn_type == "SRU":
            # SRU doesn't support PackedSequence.
            self.no_pack_padded_seq = True
            self.rnn = onmt.modules.SRU(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)
        else:
            self.rnn = getattr(nn, rnn_type)(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)

    def forward(self, input, lengths=None, hidden=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None and not self.no_pack_padded_seq:
            outputs = unpack(outputs)[0]

        return hidden_t, outputs


class RNNDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`onmt.Models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Context]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None, z_size=None):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.z_size = z_size

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type, self._input_size, hidden_size,
                                   num_layers, dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
            self._copy = True

    def forward(self, z_feature, input, context, state, context_lengths=None):
        """
        Args:
            input (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            context (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.Models.DecoderState`):
                 decoder state object to initialize the decoder
            context_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * outputs: output from the decoder
                         `[tgt_len x batch x hidden]`.
                * state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Args Check
        assert isinstance(state, RNNDecoderState)
        input_len, input_batch, _ = input.size()
        contxt_len, contxt_batch, _ = context.size()
        aeq(input_batch, contxt_batch)
        # END Args Check

        # Run the forward pass of the RNN.
        hidden, outputs, attns, coverage = self._run_forward_pass(z_feature, input, context, state, context_lengths)

        # Update the state with the result.
        final_output = outputs[-1]
        state.update_state(hidden, final_output.unsqueeze(0),
                           coverage.unsqueeze(0)
                           if coverage is not None else None)

        # Concatenates sequence of tensors along a new dimension.
        outputs = torch.stack(outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return outputs, state, attns

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        # logging.info("hidden size: {}".format(h.size()))
        #(8, 256, 500)
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        # logging.info("hidden size: {}".format(h.size()))
        #(4, 256, 1000)

        return h

    def init_decoder_state(self, src, context, enc_hidden):
        if isinstance(enc_hidden, tuple):  # LSTM
            return RNNDecoderState(context, self.hidden_size,
                                   tuple([self._fix_enc_hidden(enc_hidden[i])
                                         for i in range(len(enc_hidden))]))
        else:  # GRU
            return RNNDecoderState(context, self.hidden_size,
                                   self._fix_enc_hidden(enc_hidden))


class StdRNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """
    def _run_forward_pass(self, z_feature, input, context, state, context_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            context_lengths (LongTensor): the source context lengths.
        Returns:
            hidden (Variable): final hidden state from the decoder.
            outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
            coverage (FloatTensor, optional): coverage from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        coverage = None

        emb = self.embeddings(input)


        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, hidden = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, hidden = self.rnn(emb, state.hidden)
        # Result Check
        input_len, input_batch, _ = input.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(input_len, output_len)
        aeq(input_batch, output_batch)
        # END Result Check

        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            rnn_output.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1),                  # (contxt_len, batch, d)
            context_lengths=context_lengths
        )
        attns["std"] = attn_scores

        # Calculate the context gate.
        if self.context_gate is not None:
            outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                attn_outputs.view(-1, attn_outputs.size(2))
            )
            outputs = outputs.view(input_len, input_batch, self.hidden_size)
            outputs = self.dropout(outputs)
        else:
            outputs = self.dropout(attn_outputs)    # (input_len, batch, d)

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        """
        Private helper for building standard decoder RNN.
        """
        # Use pytorch version when available.
        if rnn_type == "SRU":
            return onmt.modules.SRU(
                    input_size, hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)

        return getattr(nn, rnn_type)(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout)

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Context n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, z_feature, input, context, state, context_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        output = state.input_feed.squeeze(0)
        output_batch, _ = output.size()
        input_len, input_batch, _ = input.size()
        aeq(input_batch, output_batch)
        # END Additional args check.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim

        print(emb.shape)
        # import pdb
        # pdb.set_trace()

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        print(z_feature.shape)
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            print(i, emb_t.shape, output.shape)
            emb_t = torch.cat([emb_t, output], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(
                rnn_output,
                context.transpose(0, 1),
                context_lengths=context_lengths)
            
            # self.context_gate is always None in our exp
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                output = self.context_gate(
                    emb_t, rnn_output, attn_output
                )
                output = self.dropout(output)
            else:
                output = self.dropout(attn_output)
            outputs += [output]
            attns["std"] += [attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + attn \
                    if coverage is not None else attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy:
                _, copy_attn = self.copy_attn(output,
                                              context.transpose(0, 1))
                attns["copy"] += [copy_attn]

        # Return result.
        raise ValueError('debug')
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


class Input_Z_FeedRNNDecoder(RNNDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Context n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, z_feature, input, context, state, context_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        output = state.input_feed.squeeze(0)
        output_batch, _ = output.size()
        input_len, input_batch, _ = input.size()
        aeq(input_batch, output_batch)
        # END Additional args check.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim

        # print(emb.shape)
        # import pdb
        # pdb.set_trace()

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        # print(z_feature.shape)
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            # print(i, emb_t.shape, output.shape, z_feature.shape)
            emb_t = torch.cat([emb_t, output, z_feature], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(
                rnn_output,
                context.transpose(0, 1),
                context_lengths=context_lengths)
            
            # self.context_gate is always None in our exp
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                output = self.context_gate(
                    emb_t, rnn_output, attn_output
                )
                output = self.dropout(output)
            else:
                output = self.dropout(attn_output)
            outputs += [output]
            attns["std"] += [attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + attn \
                    if coverage is not None else attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy:
                _, copy_attn = self.copy_attn(output,
                                              context.transpose(0, 1))
                attns["copy"] += [copy_attn]

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size + self.z_size


class Input_Z_RNNDecoder(RNNDecoderBase):

    def _run_forward_pass(self, z_feature, input, context, state, context_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        output = state.input_feed.squeeze(0)
        output_batch, _ = output.size()
        input_len, input_batch, _ = input.size()
        aeq(input_batch, output_batch)
        # END Additional args check.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim

        # print(emb.shape)
        # import pdb
        # pdb.set_trace()

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        # print(z_feature.shape)
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            # print(i, emb_t.shape, output.shape)
            emb_t = torch.cat([emb_t, z_feature], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(
                rnn_output,
                context.transpose(0, 1),
                context_lengths=context_lengths)
            
            # self.context_gate is always None in our exp
            outputs += [output]
            attns["std"] += [attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + attn \
                    if coverage is not None else attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy:
                _, copy_attn = self.copy_attn(output,
                                              context.transpose(0, 1))
                attns["copy"] += [copy_attn]

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.z_size


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, lengths)
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state,
                                             context_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        for e in self._all:
            a, br, d = e.size()
            sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, context, hidden_size, rnnstate):
        """
        Args:
            context (FloatTensor): output from the encoder of size
                                   len x batch x rnn_size.
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate (Variable): final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
            input_feed (FloatTensor): output from last layer of the decoder.
            coverage (FloatTensor): coverage output from the decoder.
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = context.size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1))
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]



class MirrorModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
      This func only support LSTM now.
    """
    def __init__(self, encoder_utt, encoder_utt_y, encoder_ctx, decoder_cxz2y, decoder_cz2x, decoder_cyz2x, decoder_cz2y,multigpu=False, opt=None):
        self.multigpu = multigpu
        super(MirrorModel, self).__init__()
        self.bidirectional_encoder = opt.brnn
        self.bidirectional_ctx_encoder = opt.ctx_bid

        self.hidden_size = opt.rnn_size
        self.encoder = encoder_utt
        # self.encoder_y = encoder_utt_y
        self.encoder_y = encoder_utt
        self.encoder_ctx = encoder_ctx
        self.decoder_cxz2y = decoder_cxz2y
        self.decoder_cz2x = decoder_cz2x
        self.decoder_cyz2x = decoder_cyz2x
        self.decoder_cz2y = decoder_cz2y
        self.encoder_state_fix_x = EncoderStateFix(opt)
        self.encoder_state_fix_y = EncoderStateFix(opt)
        self.encoder_state_fix_ctx = EncoderStateFix(opt)
        self.encoder2z = EncoderState2Z(opt)
        self.decoder_init_builder = BuildDecoderState(opt)

        # self.encoder_state_fix_xy = EncoderStateFix_test(opt)
        # self.encoder_state_fix_ctx = EncoderStateFix_test(opt)
        # self.encoder2z = EncoderState2Z_test(opt)
        # self.decoder_init_builder = BuildDecoderState_test(opt)
        self.cz_no_c=opt.cz_no_c
        # self.ctx_att_always=opt.ctx_att_always
        self.ctx_cat=opt.ctx_cat
        
    
    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        # logging.info("hidden size: {}".format(h.size()))
        #(8, 256, 500)
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        # logging.info("hidden size: {}".format(h.size()))
        #(4, 256, 1000)
        return h

    def _fix_ctx_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        # logging.info("hidden size: {}".format(h.size()))
        #(8, 256, 500)
        if self.bidirectional_ctx_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        # logging.info("hidden size: {}".format(h.size()))
        #(4, 256, 1000)
        return h
    
    def reshape_enc_hidden(self, enc_hidden, ctx_enc=False):
        if ctx_enc:
            if isinstance(enc_hidden, tuple):  # LSTM
                enc_hidden_shaped = tuple([self._fix_ctx_enc_hidden(enc_hidden[i]) for i in range(len(enc_hidden))])
            else:  # GRU
                enc_hidden_shaped = self._fix_ctx_enc_hidden(enc_hidden)
        else:
            if isinstance(enc_hidden, tuple):  # LSTM
                enc_hidden_shaped = tuple([self._fix_enc_hidden(enc_hidden[i]) for i in range(len(enc_hidden))])
            else:  # GRU
                enc_hidden_shaped = self._fix_enc_hidden(enc_hidden)
        #(4, 256, 1000)
        return enc_hidden_shaped

    def forward(self, src, tgt, ctx, lengths, src_back, tgt_back, lengths_back, lengths_ctx, dec_state_cxz2y=None, dec_state_cyz2x=None, dec_state_cz2x=None, dec_state_cz2y=None, ctx_cat=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        # self.ctx_bid = True
        tgt = tgt[:-1]  # exclude last target from inputs
        tgt_back = tgt_back[:-1]

        enc_hidden_x, context_x = self.encoder(src,)

        # enc_hidden_y, context_y = self.encoder(src_back,)
        enc_hidden_y, context_y = self.encoder_y(src_back,)


        enc_hidden_ctx, context_ctx = self.encoder_ctx(ctx)

        # logging.info("enc_hidden shape: {},{}, context shape:{}".format(enc_hidden[0].shape, enc_hidden[1].shape, context.shape))
        # (4*2, 256, 500), (4*2, 256, 500) ,   (40, 256, 1000)
        
        enc_hidden_x_shaped =  self.reshape_enc_hidden(enc_hidden_x, False)
        enc_hidden_y_shaped = self.reshape_enc_hidden(enc_hidden_y, False)
        enc_hidden_ctx_shaped = self.reshape_enc_hidden(enc_hidden_ctx, True)
        #(4, 256, 1000)

        enc_hidden_x = self.encoder_state_fix_x(enc_hidden_x_shaped)
        enc_hidden_y = self.encoder_state_fix_y(enc_hidden_y_shaped)

        # enc_hidden_x = self.encoder_state_fix_xy(enc_hidden_x_shaped)
        # enc_hidden_y = self.encoder_state_fix_xy(enc_hidden_y_shaped)

        enc_hidden_ctx = self.encoder_state_fix_ctx(enc_hidden_ctx_shaped)
        # print(enc_hidden_x.shape)
        # print(enc_hidden_y.shape)
        # print(enc_hidden_ctx.shape)
        # (256, 1000)
        # test: (256, rnn=1000)

        enc_hidden_cxy = torch.cat([enc_hidden_ctx, enc_hidden_x, enc_hidden_y], -1)
        # (256, 3000)
        
        # 132, 133: no detach, enc_hidden_cz = enc_hidden_ctx
        # 134, 135: detach, enc_hidden_cz = enc_hidden_ctx
        # 136, 137: detach,  enc_hidden_cz = torch.cat([enc_hidden_ctx, rec_z], -1)

        rec_z, rec_mu, rec_logvar = self.encoder2z.cxy2z(enc_hidden_cxy)
        prior_z, prior_mu, prior_logvar = self.encoder2z.c2z(enc_hidden_ctx)
        real_z = rec_z
        # (256, z_dim)

        # enc_hidden_cxz = torch.cat([enc_hidden_ctx, enc_hidden_x, rec_z], -1)
        # enc_hidden_cyz = torch.cat([enc_hidden_ctx, enc_hidden_y, rec_z], -1)
        # if not self.cz_no_c:    # default
        #     enc_hidden_cz = torch.cat([enc_hidden_ctx, rec_z], -1)
        # else:
        #     enc_hidden_cz = rec_z

        enc_hidden_cxz = torch.cat([enc_hidden_ctx, enc_hidden_x], -1)
        enc_hidden_cyz = torch.cat([enc_hidden_ctx, enc_hidden_y], -1)
        enc_hidden_cz = enc_hidden_ctx
        # print(len(enc_hidden_cz))
        # print(enc_hidden_cz.shape)
        

        enc_hidden_cxz, enc_hidden_cyz, enc_hidden_cz = self.decoder_init_builder(enc_hidden_cxz, enc_hidden_cyz, enc_hidden_cz)
        
        # print(context_x.shape)
        # print(context_y.shape)
        # print(context_ctx.shape)
        # print(lengths.shape)
        # print(lengths)
        # print(lengths_back.shape)
        # print(lengths_back)
        # print(lengths_ctx.shape)
        # print(lengths_ctx)
        
        if not self.ctx_cat:
            enc_state_cxz = RNNDecoderState(context_x, self.hidden_size, enc_hidden_cxz)
            enc_state_cyz = RNNDecoderState(context_y, self.hidden_size, enc_hidden_cyz)
            enc_state_cz = RNNDecoderState(context_ctx, self.hidden_size, enc_hidden_cz)
            out_cxz2y, dec_state_cxz2y, attns_cxz2y = self.decoder_cxz2y(real_z, tgt, context_x,
                                            enc_state_cxz if dec_state_cxz2y is None else dec_state_cxz2y,
                                            context_lengths=lengths)

            out_cyz2x, dec_state_cyz2x, attns_cyz2x = self.decoder_cyz2x(real_z,tgt_back, context_y,
                                            enc_state_cyz if dec_state_cyz2x is None else dec_state_cyz2x,
                                            context_lengths=lengths_back)

        else:
            #################################################
            context_ctx_x = torch.cat([context_ctx, context_x], 0)
            context_ctx_y = torch.cat([context_ctx, context_y], 0)
            # lengths_ctx_x = lengths_ctx + lengths
            # lengths_ctx_y = lengths_ctx + lengths_back

            enc_state_cxz = RNNDecoderState(context_ctx_x, self.hidden_size, enc_hidden_cxz)
            enc_state_cyz = RNNDecoderState(context_ctx_y, self.hidden_size, enc_hidden_cyz)
            enc_state_cz = RNNDecoderState(context_ctx, self.hidden_size, enc_hidden_cz)

            out_cxz2y, dec_state_cxz2y, attns_cxz2y = self.decoder_cxz2y(real_z, tgt, context_ctx_x, enc_state_cxz if dec_state_cxz2y is None else dec_state_cxz2y)

            out_cyz2x, dec_state_cyz2x, attns_cyz2x = self.decoder_cyz2x(real_z, tgt_back, context_ctx_y, enc_state_cyz if dec_state_cyz2x is None else dec_state_cyz2x)
            ######################################################


        out_cz2x, dec_state_cz2x, attns_cz2x = self.decoder_cz2x(real_z,tgt_back, context_ctx,
                                        enc_state_cz if dec_state_cz2x is None else dec_state_cz2x,
                                        context_lengths=lengths_ctx)

        out_cz2y, dec_state_cz2y, attns_cz2y = self.decoder_cz2y(real_z,tgt, context_ctx,
                                            enc_state_cz if dec_state_cz2y is None else dec_state_cz2y,
                                            context_lengths=lengths_ctx)
        kl_loss = self.encoder2z.kl_loss(rec_mu, rec_logvar, prior_mu, prior_logvar)
        results = Pack(
                        out_cxz2y=out_cxz2y, dec_state_cxz2y=dec_state_cxz2y, attns_cxz2y=attns_cxz2y,
                        out_cyz2x=out_cyz2x, dec_state_cyz2x=dec_state_cyz2x, attns_cyz2x=attns_cyz2x,
                        out_cz2x=out_cz2x, dec_state_cz2x=dec_state_cz2x, attns_cz2x=attns_cz2x,
                        out_cz2y=out_cz2y, dec_state_cz2y=dec_state_cz2y, attns_cz2y=attns_cz2y,
                        kl_loss=kl_loss.sum()
                       )

        return results
    
    def forward_beam(self, src, ctx, lengths, lengths_ctx, ctx_cat=None):
        enc_hidden_x, context_x = self.encoder(src,)
        # enc_hidden_y, context_y = self.encoder(src_back, lengths_back)
        # enc_hidden_ctx, context_ctx = self.encoder_ctx(ctx, lengths_ctx)
        enc_hidden_ctx, context_ctx = self.encoder_ctx(ctx)
        context_ctx_x = torch.cat([context_ctx, context_x], 0)

        # logging.info("enc_hidden shape: {},{}, context shape:{}".format(enc_hidden[0].shape, enc_hidden[1].shape, context.shape))
        # (4*2, 256, 500), (4*2, 256, 500) ,   (40, 256, 1000)
        enc_hidden_x_shaped =  self.reshape_enc_hidden(enc_hidden_x, False)
        enc_hidden_ctx_shaped = self.reshape_enc_hidden(enc_hidden_ctx, True)
        #(4, 256, 1000)
        enc_hidden_x = self.encoder_state_fix_x(enc_hidden_x_shaped)
        enc_hidden_ctx = self.encoder_state_fix_ctx(enc_hidden_ctx_shaped)
        # (256, 800)
        prior_z, prior_mu, prior_logvar = self.encoder2z.c2z(enc_hidden_ctx)
        # (256, z_dim)
        # enc_hidden_cxz = torch.cat([enc_hidden_ctx, enc_hidden_x, prior_z], -1)
        enc_hidden_cxz = torch.cat([enc_hidden_ctx, enc_hidden_x], -1)
        enc_hidden_cxz = self.decoder_init_builder.forward_single(enc_hidden_cxz)
        if self.ctx_cat:
            enc_state_cxz = RNNDecoderState(context_ctx_x, self.hidden_size, enc_hidden_cxz)
            return enc_state_cxz, context_ctx_x, prior_z
        else:
            enc_state_cxz = RNNDecoderState(context_x, self.hidden_size, enc_hidden_cxz)
            return enc_state_cxz, context_x, prior_z
    
    def decode_beam(self,prior_z, tgt, context_x, dec_state_cxz2y, lengths, ctx_cat=None):
        # ctx_cat = self.ctx_cat
        if self.ctx_cat:
            out_cxz2y, dec_state_cxz2y, attns_cxz2y = self.decoder_cxz2y(prior_z, tgt, context_x, dec_state_cxz2y)
        else:
            out_cxz2y, dec_state_cxz2y, attns_cxz2y = self.decoder_cxz2y(prior_z, tgt, context_x, dec_state_cxz2y, context_lengths=lengths)
        return out_cxz2y, dec_state_cxz2y, attns_cxz2y

        
class EncoderStateFix(nn.Module):
    """
    """
    def __init__(self,opt):
        super(EncoderStateFix, self).__init__()
        enc_layers = opt.enc_layers
        self.encoder2z_layer1 = nn.Sequential(
                                    nn.Linear(opt.rnn_size * enc_layers, opt.rnn_size),
                                    nn.ReLU()
                                )
        # self.encoder2z_layer1 = nn.Linear(opt.rnn_size * enc_layers, opt.rnn_size)
    def forward(self, encoder_hidden):
        #(4, 256, 1000)
        # return encoder_hidden[-1]
        batch_size = encoder_hidden.size(1)
        return self.encoder2z_layer1(encoder_hidden.transpose(0, 1).contiguous().view(batch_size, -1))


class EncoderState2Z(nn.Module):
    """
    """
    def __init__(self,opt):
        super(EncoderState2Z, self).__init__()
        enc_size = opt.rnn_size 
        z_dim = opt.z_dim
        self.cxy2z_layer = nn.Sequential(
                nn.Linear(enc_size * 3, z_dim * 2),
                nn.ReLU()
            )
        self.cxy2z_mu = nn.Linear(z_dim * 2, z_dim)
        self.cxy2z_logvar = nn.Linear(z_dim * 2, z_dim)
        # Prior Net
        self.c2z_layer = nn.Sequential(
            nn.Linear(enc_size, z_dim * 2),
            nn.ReLU()
        )
        self.c2z_mu = nn.Linear(z_dim * 2, z_dim)
        self.c2z_logvar = nn.Linear(z_dim * 2, z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = Variable(torch.rand(std.size()).cuda())
        # eps = torch.rand(std.size())
        # eps = torch.rand(std.size()).cuda()
        return mu + std * eps

    def kl_loss(self, mu1, logvar1, mu2, logvar2):
        # mu1, logvar1 -> RecognitionNet
        # mu2, logvar2 -> PriorNet
        kld = -0.5 * torch.sum(1 + logvar1 - logvar2 - torch.exp(logvar1)/torch.exp(logvar2) - torch.pow(mu1 - mu2, 2)/torch.exp(logvar2),-1)
        return kld
    def kl_loss_gaussian(self, mu1, logvar1):
        # mu2=0, logvar2=0
        kld = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(),-1)
        return kld

    def cxy2z(self, hidden):
        hidden_ = self.cxy2z_layer(hidden)
        mu = self.cxy2z_mu(hidden_)
        logvar = self.cxy2z_logvar(hidden_)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def c2z(self, hidden):
        hidden_ = self.c2z_layer(hidden)
        mu = self.c2z_mu(hidden_)
        logvar = self.c2z_logvar(hidden_)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class BuildDecoderState(nn.Module):
    """
    """
    def __init__(self,opt):
        super(BuildDecoderState, self).__init__()
        enc_size = opt.rnn_size
        decoder_size = opt.dec_layers * opt.rnn_size
        self.rnn_size, self.num_layer = opt.rnn_size, opt.dec_layers
        self.rnn_type=opt.rnn_type
        self.cz_no_c=opt.cz_no_c
        if opt.rnn_type=='GRU':
            # self.cxz2y = nn.Linear(enc_size * 2, decoder_size)
            # self.cyz2x = nn.Linear(enc_size * 2, decoder_size)
            # self.cz2xy = nn.Linear(enc_size, decoder_size)

            self.cxz2y = nn.Sequential(nn.Linear(enc_size * 2, decoder_size), nn.ReLU())
            self.cyz2x = nn.Sequential(nn.Linear(enc_size * 2, decoder_size), nn.ReLU())
            self.cz2xy = nn.Sequential(nn.Linear(enc_size, decoder_size), nn.ReLU())
        else:
            raise ValueError
    
    def forward(self, cxz, cyz, cz):
        # (1*256*2000, 1*256*2000, 1*256*1000)
        if self.rnn_type=='GRU':
            cxz_ = self.cxz2y(cxz)
            cyz_ = self.cyz2x(cyz)
            cz_ = self.cz2xy(cz)
            # 1*256*4000
            cxz_ = cxz_.view(-1, self.num_layer, self.rnn_size).transpose(0,1)
            cyz_ = cyz_.view(-1, self.num_layer, self.rnn_size).transpose(0,1)
            cz_ = cz_.view(-1, self.num_layer, self.rnn_size).transpose(0,1)
            # (4, 256, 1000)
        else:
            raise ValueError("the rnn type: {} is not supported".format(opt.rnn_type))
        return cxz_, cyz_, cz_

    def forward_single(self, cxz):
        if self.rnn_type=='GRU':
            cxz_ = self.cxz2y(cxz)
            cxz_ = cxz_.view(-1, self.num_layer, self.rnn_size).transpose(0,1)
            # (4, 256, 1000)
        else:
            raise ValueError("the rnn type: {} is not supported".format(opt.rnn_type))
        return cxz_


        
        
#### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  #####

class MirrorLight(MirrorModel):
    def __init__(self, encoder_utt, encoder_utt_y, encoder_ctx, decoder_cxz2y, decoder_cz2x, decoder_cyz2x, decoder_cz2y,multigpu=False, opt=None):
        super(MirrorLight, self).__init__(encoder_utt, encoder_utt_y, encoder_ctx, decoder_cxz2y, decoder_cz2x, decoder_cyz2x, decoder_cz2y,multigpu=multigpu, opt=opt)

    def forward(self, src, tgt, ctx, lengths, src_back, tgt_back, lengths_back, lengths_ctx, dec_state_cxz2y=None, dec_state_cyz2x=None, dec_state_cz2x=None, dec_state_cz2y=None, ctx_cat=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        # self.ctx_bid = True
        tgt = tgt[:-1]  # exclude last target from inputs
        tgt_back = tgt_back[:-1]

        enc_hidden_x, context_x = self.encoder(src,)

        # enc_hidden_y, context_y = self.encoder(src_back,)
        enc_hidden_y, context_y = self.encoder_y(src_back,)


        enc_hidden_ctx, context_ctx = self.encoder_ctx(ctx)

        # logging.info("enc_hidden shape: {},{}, context shape:{}".format(enc_hidden[0].shape, enc_hidden[1].shape, context.shape))
        # (4*2, 256, 500), (4*2, 256, 500) ,   (40, 256, 1000)
        
        enc_hidden_x_shaped =  self.reshape_enc_hidden(enc_hidden_x, False)
        enc_hidden_y_shaped = self.reshape_enc_hidden(enc_hidden_y, False)
        enc_hidden_ctx_shaped = self.reshape_enc_hidden(enc_hidden_ctx, True)
        #(4, 256, 1000)

        enc_hidden_x = self.encoder_state_fix_x(enc_hidden_x_shaped)
        enc_hidden_y = self.encoder_state_fix_y(enc_hidden_y_shaped)

        # enc_hidden_x = self.encoder_state_fix_xy(enc_hidden_x_shaped)
        # enc_hidden_y = self.encoder_state_fix_xy(enc_hidden_y_shaped)

        enc_hidden_ctx = self.encoder_state_fix_ctx(enc_hidden_ctx_shaped)
        # print(enc_hidden_x.shape)
        # print(enc_hidden_y.shape)
        # print(enc_hidden_ctx.shape)
        # (256, 1000)
        # test: (256, rnn=1000)

        enc_hidden_cxy = torch.cat([enc_hidden_ctx, enc_hidden_x, enc_hidden_y], -1)
        # (256, 3000)
        
        # 132, 133: no detach, enc_hidden_cz = enc_hidden_ctx
        # 134, 135: detach, enc_hidden_cz = enc_hidden_ctx
        # 136, 137: detach,  enc_hidden_cz = torch.cat([enc_hidden_ctx, rec_z], -1)

        rec_z, rec_mu, rec_logvar = self.encoder2z.c2z(enc_hidden_ctx)  # this is the light version: p(z|c) is the posterior estimation
        real_z = rec_z

        # enc_hidden_cxz = torch.cat([enc_hidden_ctx, enc_hidden_x, rec_z], -1)
        # enc_hidden_cyz = torch.cat([enc_hidden_ctx, enc_hidden_y, rec_z], -1)
        # if not self.cz_no_c:    # default
        #     enc_hidden_cz = torch.cat([enc_hidden_ctx, rec_z], -1)
        # else:
        #     enc_hidden_cz = rec_z

        enc_hidden_cxz = torch.cat([enc_hidden_ctx, enc_hidden_x], -1)
        enc_hidden_cyz = torch.cat([enc_hidden_ctx, enc_hidden_y], -1)
        enc_hidden_cz = enc_hidden_ctx
        # print(len(enc_hidden_cz))
        # print(enc_hidden_cz.shape)
        

        enc_hidden_cxz, enc_hidden_cyz, enc_hidden_cz = self.decoder_init_builder(enc_hidden_cxz, enc_hidden_cyz, enc_hidden_cz)
        
        
        if not self.ctx_cat:
            enc_state_cxz = RNNDecoderState(context_x, self.hidden_size, enc_hidden_cxz)
            enc_state_cyz = RNNDecoderState(context_y, self.hidden_size, enc_hidden_cyz)
            enc_state_cz = RNNDecoderState(context_ctx, self.hidden_size, enc_hidden_cz)
            out_cxz2y, dec_state_cxz2y, attns_cxz2y = self.decoder_cxz2y(real_z, tgt, context_x,
                                            enc_state_cxz if dec_state_cxz2y is None else dec_state_cxz2y,
                                            context_lengths=lengths)

            out_cyz2x, dec_state_cyz2x, attns_cyz2x = self.decoder_cyz2x(real_z,tgt_back, context_y,
                                            enc_state_cyz if dec_state_cyz2x is None else dec_state_cyz2x,
                                            context_lengths=lengths_back)

        else:
            #################################################
            context_ctx_x = torch.cat([context_ctx, context_x], 0)
            context_ctx_y = torch.cat([context_ctx, context_y], 0)
            # lengths_ctx_x = lengths_ctx + lengths
            # lengths_ctx_y = lengths_ctx + lengths_back

            enc_state_cxz = RNNDecoderState(context_ctx_x, self.hidden_size, enc_hidden_cxz)
            enc_state_cyz = RNNDecoderState(context_ctx_y, self.hidden_size, enc_hidden_cyz)
            enc_state_cz = RNNDecoderState(context_ctx, self.hidden_size, enc_hidden_cz)

            out_cxz2y, dec_state_cxz2y, attns_cxz2y = self.decoder_cxz2y(real_z, tgt, context_ctx_x, enc_state_cxz if dec_state_cxz2y is None else dec_state_cxz2y)

            out_cyz2x, dec_state_cyz2x, attns_cyz2x = self.decoder_cyz2x(real_z, tgt_back, context_ctx_y, enc_state_cyz if dec_state_cyz2x is None else dec_state_cyz2x)
            ######################################################


        out_cz2x, dec_state_cz2x, attns_cz2x = self.decoder_cz2x(real_z,tgt_back, context_ctx,
                                        enc_state_cz if dec_state_cz2x is None else dec_state_cz2x,
                                        context_lengths=lengths_ctx)

        out_cz2y, dec_state_cz2y, attns_cz2y = self.decoder_cz2y(real_z,tgt, context_ctx,
                                            enc_state_cz if dec_state_cz2y is None else dec_state_cz2y,
                                            context_lengths=lengths_ctx)

        kl_loss = self.encoder2z.kl_loss_gaussian(rec_mu, rec_logvar)
        results = Pack(
                        out_cxz2y=out_cxz2y, dec_state_cxz2y=dec_state_cxz2y, attns_cxz2y=attns_cxz2y,
                        out_cyz2x=out_cyz2x, dec_state_cyz2x=dec_state_cyz2x, attns_cyz2x=attns_cyz2x,
                        out_cz2x=out_cz2x, dec_state_cz2x=dec_state_cz2x, attns_cz2x=attns_cz2x,
                        out_cz2y=out_cz2y, dec_state_cz2y=dec_state_cz2y, attns_cz2y=attns_cz2y,
                        kl_loss=kl_loss.sum()
                       )

        return results
    
    def forward_beam(self, src, ctx, lengths, lengths_ctx, ctx_cat=None):
        enc_hidden_x, context_x = self.encoder(src,)
        # enc_hidden_y, context_y = self.encoder(src_back, lengths_back)
        # enc_hidden_ctx, context_ctx = self.encoder_ctx(ctx, lengths_ctx)
        enc_hidden_ctx, context_ctx = self.encoder_ctx(ctx)
        context_ctx_x = torch.cat([context_ctx, context_x], 0)

        # logging.info("enc_hidden shape: {},{}, context shape:{}".format(enc_hidden[0].shape, enc_hidden[1].shape, context.shape))
        # (4*2, 256, 500), (4*2, 256, 500) ,   (40, 256, 1000)
        enc_hidden_x_shaped =  self.reshape_enc_hidden(enc_hidden_x, False)
        enc_hidden_ctx_shaped = self.reshape_enc_hidden(enc_hidden_ctx, True)
        #(4, 256, 1000)
        enc_hidden_x = self.encoder_state_fix_x(enc_hidden_x_shaped)
        enc_hidden_ctx = self.encoder_state_fix_ctx(enc_hidden_ctx_shaped)
        # (256, 800)
        rec_z, rec_mu, rec_logvar = self.encoder2z.c2z(enc_hidden_ctx)  # this is the light version: p(z|c) is the posterior estimation

        # (256, z_dim)
        # enc_hidden_cxz = torch.cat([enc_hidden_ctx, enc_hidden_x, prior_z], -1)
        enc_hidden_cxz = torch.cat([enc_hidden_ctx, enc_hidden_x], -1)
        enc_hidden_cxz = self.decoder_init_builder.forward_single(enc_hidden_cxz)
        if self.ctx_cat:
            enc_state_cxz = RNNDecoderState(context_ctx_x, self.hidden_size, enc_hidden_cxz)
            return enc_state_cxz, context_ctx_x, rec_z
        else:
            enc_state_cxz = RNNDecoderState(context_x, self.hidden_size, enc_hidden_cxz)
            return enc_state_cxz, context_x, rec_z

class EncoderState2zLow(nn.Module):
    """
    """
    def __init__(self,opt):
        super(EncoderState2zLow, self).__init__()
        enc_size = opt.enc_layers * 200
        z_dim = opt.z_dim
        self.cxy2z_layer = nn.Sequential(
                nn.Linear(enc_size * 3, z_dim * 2),
                nn.ReLU()
            )
        self.cxy2z_mu = nn.Linear(z_dim * 2, z_dim)
        self.cxy2z_logvar = nn.Linear(z_dim * 2, z_dim)
        # Prior Net
        self.cx2z_layer = nn.Sequential(
            nn.Linear(enc_size * 2, z_dim * 2),
            nn.ReLU()
        )
        self.cx2z_mu = nn.Linear(z_dim * 2, z_dim)
        self.cx2z_logvar = nn.Linear(z_dim * 2, z_dim)


        self.cy2z_layer = nn.Sequential(
            nn.Linear(enc_size * 2, z_dim * 2),
            nn.ReLU()
        )
        self.cy2z_mu = nn.Linear(z_dim * 2, z_dim)
        self.cy2z_logvar = nn.Linear(z_dim * 2, z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = Variable(torch.rand(std.size()).cuda())
        # eps = torch.rand(std.size())
        # eps = torch.rand(std.size()).cuda()
        return mu + std * eps

    def kl_loss(self, mu1, logvar1, mu2, logvar2):
        # mu1, logvar1 -> RecognitionNet
        # mu2, logvar2 -> PriorNet
        kld = -0.5 * torch.sum(1 + logvar1 - logvar2 - torch.exp(logvar1)/torch.exp(logvar2) - torch.pow(mu1 - mu2, 2)/torch.exp(logvar2),-1)
        return kld
    def kl_loss_gaussian(self, mu1, logvar1):
        # mu2=0, logvar2=0
        kld = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(),-1)
        return kld

    def cxy2z(self, hidden):
        hidden_ = self.cxy2z_layer(hidden)
        mu = self.cxy2z_mu(hidden_)
        logvar = self.cxy2z_logvar(hidden_)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def cx2z(self, hidden):
        hidden_ = self.cx2z_layer(hidden)
        mu = self.cx2z_mu(hidden_)
        logvar = self.cx2z_logvar(hidden_)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def cy2z(self, hidden):
        hidden_ = self.cy2z_layer(hidden)
        mu = self.cy2z_mu(hidden_)
        logvar = self.cy2z_logvar(hidden_)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Mirror_Encshare(MirrorModel):
    def __init__(self, encoder_utt, decoder_cxz2y, decoder_cz2x, decoder_cyz2x, decoder_cz2y,multigpu=False, opt=None):
        super(Mirror_Encshare, self).__init__(encoder_utt, None, decoder_cxz2y, decoder_cz2x, decoder_cyz2x, decoder_cz2y, multigpu, opt)
        self.encoder_ctx = encoder_utt


class MirrorLow_Encshare(MirrorLight):
    def __init__(self, encoder_utt,  decoder_cxz2y, decoder_cz2x, decoder_cyz2x, decoder_cz2y,multigpu=False, opt=None):
        super(MirrorLow_Encshare, self).__init__(encoder_utt, None, decoder_cxz2y, decoder_cz2x, decoder_cyz2x, decoder_cz2y, multigpu, opt)
        self.encoder_ctx = encoder_utt





########################## TEST ##########################
      
class EncoderStateFix_test(nn.Module):
    """
      This func only support LSTM now.
    """
    def __init__(self,opt):
        super(EncoderStateFix_test, self).__init__()
        
        if opt.rnn_type=='LSTM':
            self.encoder2z_layer1_h = nn.Sequential(
                nn.Linear(opt.rnn_size, opt.rnn_size//2),
                nn.ReLU()
            )
            self.encoder2z_layer1_c = nn.Sequential(
                nn.Linear(opt.rnn_size, opt.rnn_size//2),
                nn.ReLU()
            )
        else:
            self.encoder2z_layer1_h = nn.Sequential(
                nn.Linear(opt.rnn_size, opt.rnn_size),
                nn.ReLU()
            )

    def forward(self, encoder_hidden):
        #(4, 256, 1000)
        # (1, 256, 1000)
        if isinstance(encoder_hidden, tuple):
            batch_size = encoder_hidden[0].size(1)
            h1 = self.encoder2z_layer1_h(encoder_hidden[0][-1].unsqueeze(0))
            c1 = self.encoder2z_layer1_c(encoder_hidden[1][-1].unsqueeze(0))
            h1 = h1.transpose(0,1).contiguous().view(batch_size, -1)
            c1 = c1.transpose(0,1).contiguous().view(batch_size, -1)
            hc = torch.cat([h1, c1], -1)
            # 256, 1000
            # 1000 = 1 * 500 * 2
        else:
            batch_size = encoder_hidden.size(1)
            h1 = self.encoder2z_layer1_h(encoder_hidden[-1].unsqueeze(0))
            hc = h1.transpose(0,1).contiguous().view(batch_size, -1)
            # 256, 1000
            # 1000 = 1 * 1000 
        return hc 

class EncoderState2Z_test(nn.Module):
    """
    """
    def __init__(self,opt):
        super(EncoderState2Z_test, self).__init__()
        enc_size = opt.rnn_size
        z_dim = opt.z_dim
        self.cxy2z_layer = nn.Sequential(
                nn.Linear(enc_size * 3, z_dim * 2),
                nn.ReLU()
            )
        self.cxy2z_mu = nn.Linear(z_dim * 2, z_dim)
        self.cxy2z_logvar = nn.Linear(z_dim * 2, z_dim)
        # Prior Net
        self.c2z_layer = nn.Sequential(
            nn.Linear(enc_size, z_dim * 2),
            nn.ReLU()
        )
        self.c2z_mu = nn.Linear(z_dim * 2, z_dim)
        self.c2z_logvar = nn.Linear(z_dim * 2, z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = Variable(torch.rand(std.size()).cuda())
        # eps = torch.rand(std.size())
        # eps = torch.rand(std.size()).cuda()
        return mu + std * eps

    def kl_loss(self, mu1, logvar1, mu2, logvar2):
        # mu1, logvar1 -> RecognitionNet
        # mu2, logvar2 -> PriorNet
        kld = -0.5 * torch.sum(1 + logvar1 - logvar2 - torch.exp(logvar1)/torch.exp(logvar2) - torch.pow(mu1 - mu2, 2)/torch.exp(logvar2),-1)
        return kld
    def kl_loss_gaussian(self, mu1, logvar1):
        # mu2=0, logvar2=0
        kld = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(),-1)
        return kld

    def cxy2z(self, hidden):
        hidden_ = self.cxy2z_layer(hidden)
        mu = self.cxy2z_mu(hidden_)
        logvar = self.cxy2z_logvar(hidden_)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def c2z(self, hidden):
        hidden_ = self.c2z_layer(hidden)
        mu = self.c2z_mu(hidden_)
        logvar = self.c2z_logvar(hidden_)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class BuildDecoderState_test(nn.Module):
    """
    """
    def __init__(self,opt):
        super(BuildDecoderState_test, self).__init__()
        enc_size = opt.rnn_size
        z_dim = opt.z_dim
        decoder_size = opt.dec_layers * opt.rnn_size
        self.rnn_size, self.num_layer = opt.rnn_size, opt.dec_layers
        self.rnn_type=opt.rnn_type
        self.cz_no_c=opt.cz_no_c
        if opt.rnn_type=='GRU':
            self.cxz2y = nn.Sequential(
                    nn.Linear(enc_size * 2 + z_dim, decoder_size),
                    nn.ReLU()
                )
            # 256 * 1700 -> 256 * 4000
            self.cyz2x = nn.Sequential(
                    nn.Linear(enc_size * 2 + z_dim, decoder_size),
                    nn.ReLU()
                )
            self.cz2xy = nn.Sequential(
                    nn.Linear(enc_size + z_dim, decoder_size),
                    nn.ReLU()
                )
        else:
            self.cxz2y_h = nn.Sequential(
                    nn.Linear(enc_size * 2 + z_dim, decoder_size),
                    nn.ReLU()
                )
            # 256 * 1700 -> 256 * 4000
            self.cyz2x_h = nn.Sequential(
                    nn.Linear(enc_size * 2 + z_dim, decoder_size),
                    nn.ReLU()
                )
            if not self.cz_no_c:
                self.cz2xy_h = nn.Sequential(
                        nn.Linear(enc_size + z_dim, decoder_size),
                        nn.ReLU()
                    )
            else:
                self.cz2xy_h = nn.Sequential(
                        nn.Linear(z_dim, decoder_size),
                        nn.ReLU()
                    )
            self.cxz2y_c = nn.Sequential(
                    nn.Linear(enc_size * 2 + z_dim, decoder_size),
                    nn.ReLU()
                )
            # 256 * 1700 -> 256 * 4000
            self.cyz2x_c = nn.Sequential(
                    nn.Linear(enc_size * 2 + z_dim, decoder_size),
                    nn.ReLU()
                )
            if not self.cz_no_c:
                self.cz2xy_c = nn.Sequential(
                        nn.Linear(enc_size + z_dim, decoder_size),
                        nn.ReLU()
                    )
            else:
                self.cz2xy_c = nn.Sequential(
                        nn.Linear(z_dim, decoder_size),
                        nn.ReLU()
                    )
    
    def forward(self, cxz, cyz, cz):
        if self.rnn_type=='GRU':
            cxz_, cyz_, cz_ = self.cxz2y(cxz), self.cyz2x(cyz), self.cz2xy(cz)
            cxz_ = cxz_.view(-1, self.num_layer, self.rnn_size).transpose(0,1).contiguous()
            cyz_ = cyz_.view(-1, self.num_layer, self.rnn_size).transpose(0,1).contiguous()
            cz_ = cz_.view(-1, self.num_layer, self.rnn_size).transpose(0,1).contiguous()
            # (4, 256, 1000)
        elif self.rnn_type=='LSTM':
            cxz_h, cyz_h, cz_h = self.cxz2y_h(cxz), self.cyz2x_h(cyz), self.cz2xy_h(cz)
            cxz_c, cyz_c, cz_c = self.cxz2y_c(cxz), self.cyz2x_c(cyz), self.cz2xy_c(cz)
            cxz_h = cxz_h.view(-1, self.num_layer, self.rnn_size).transpose(0,1).contiguous()
            cyz_h = cyz_h.view(-1, self.num_layer, self.rnn_size).transpose(0,1).contiguous()
            cz_h = cz_h.view(-1, self.num_layer, self.rnn_size).transpose(0,1).contiguous()
            cxz_c = cxz_c.view(-1, self.num_layer, self.rnn_size).transpose(0,1).contiguous()
            cyz_c = cyz_c.view(-1, self.num_layer, self.rnn_size).transpose(0,1).contiguous()
            cz_c = cz_c.view(-1, self.num_layer, self.rnn_size).transpose(0,1).contiguous()
            cxz_ = tuple([cxz_h, cxz_c])
            cyz_ = tuple([cyz_h, cyz_c])
            cz_ = tuple([cz_h, cz_c])
        else:
            raise ValueError("the rnn type: {} is not supported".format(opt.rnn_type))
        return cxz_, cyz_, cz_

    def forward_single(self, cxz):
        if self.rnn_type=='GRU':
            cxz_ = self.cxz2y(cxz)
            cxz_ = cxz_.view(-1, self.num_layer, self.rnn_size).transpose(0,1).contiguous()
            # (4, 256, 1000)
        elif self.rnn_type=='LSTM':
            cxz_h = self.cxz2y_h(cxz)
            cxz_c = self.cxz2y_c(cxz)
            cxz_h = cxz_h.view(-1, self.num_layer, self.rnn_size).transpose(0,1).contiguous()
            cxz_c = cxz_c.view(-1, self.num_layer, self.rnn_size).transpose(0,1).contiguous()
            cxz_ = tuple([cxz_h, cxz_c])
        else:
            raise ValueError("the rnn type: {} is not supported".format(opt.rnn_type))
        return cxz_


   