import json
import random
import numpy as np
import csv

from numpy.lib.utils import source

def check_file(path, dest):
    with open(path, 'r') as f:
        data = f.readlines()
    new_data = []
    for line in data:
        if len(line.strip().split())<1:
            new_data.append('<unk>')
        else:
            new_data.append(line.strip())
    with open(dest, 'w') as f:
        for x in new_data:
            f.write(x+ '\n')


class build_dialogs():
    def __init__(self, ctx_path, x_path, y_path):
        self.ctx_data = self.read_files(ctx_path)
        self.x_data = self.read_files(x_path)
        self.y_data = self.read_files(y_path)
        assert len(self.ctx_data)==len(self.x_data) and len(self.ctx_data)==len(self.y_data) 

    def read_files(self, path):
        with open(path, 'r') as f:
            data = f.readlines()
        return data
    
    def build_single(self, source, dest):
        data = self.read_files(source)
        new_dialogs = []
        for idx, line in enumerate(data):
            new_dialogs.append('*********** Sampled Dialogs ***********')
            new_dialogs.append('CTX: ' + self.ctx_data[idx].strip())
            new_dialogs.append('Turn A: ' + self.x_data[idx].strip())
            new_dialogs.append('True B: ' + self.y_data[idx].strip())
            new_dialogs.append('Pred B: ' + line.strip())

        with open(dest, 'w') as f:
            for x in new_dialogs:
                f.write(x+ '\n')

    def build_multiple(self, source_dict, dest):
        # source_dict = {'a':'path1', 'b':'path2'}
        data = {}
        num = 99999
        for k,v in source_dict.items():
            data[k]=self.read_files(v)
            num = min(num, len(data[k]))
        new_dialogs = []
        for idx in range(num):
            new_dialogs.append('*********** Sampled Dialogs ***********')
            new_dialogs.append('CTX: ' + self.ctx_data[idx].strip())
            new_dialogs.append('Turn A: ' + self.x_data[idx].strip())
            new_dialogs.append('Reference: ' + self.y_data[idx].strip())
            for k, v in data.items():
                new_dialogs.append(k+': ' + v[idx].strip())
        with open(dest, 'w') as f:
            for x in new_dialogs:
                f.write(x+ '\n')

    def build_crowdsourcing_file(self, f1_path, f1_name, f2_path, f2_name, target_path):
        dialog_list = []
    
        file1 = self.read_files(f1_path)
        file2 = self.read_files(f2_path)
        print(len(file1))
        print(len(file2))
        length = min(len(file1), len(file2))
        selected_list = random.sample(range(length), 100)
        reply_order = []

        for idx in selected_list:
            context = 'Speaker A: ' + self.ctx_data[idx].strip()
            query = 'Speaker B: ' + self.x_data[idx].strip()
            ctx = context + ' <br> ' + query
            truth = 'Truth: ' + self.y_data[idx]
            r1 = file1[idx].strip()
            r2 = file2[idx].strip()

        
            if np.random.random()>0.5:
                pair_name = f1_name + '_' + f2_name
                reply1 = r1
                reply2 = r2
            else:
                pair_name = f2_name + '_' + f1_name
                reply1 = r2
                reply2 = r1
            dialog_single = {'context':ctx.replace('<unk>','_unk_'),  'truth':truth.replace('<unk>','_unk_'), 'reply_1':reply1.replace('<unk>','_unk_'), 'reply_2':reply2.replace('<unk>','_unk_'), 'order':pair_name, 'dialog_id':idx}
            dialog_list.append(dialog_single)
        with open(target_path, 'w') as f:
            fnames = ['context','truth', 'reply_1', 'reply_2', 'order', 'dialog_id']
            writer = csv.DictWriter(f, fieldnames=fnames) 
            writer.writeheader()
            writer.writerows(dialog_list)


def build_bert_files():

    def build_bert_file_step(ctx_path, y_path, tgt_bert):
        with open(ctx_path, 'r') as f1:
            ctx_data = f1.readlines()
        with open(y_path, 'r') as f2:
            y_data = f2.readlines()
        print(len(ctx_data), len(y_data))
        with open(tgt_bert, 'w') as tgt_bert:
            for idx, line in enumerate(y_data[:-2]):
                ctx_line = ctx_data[idx].replace('__utt__', '__eou__').strip()
                tgt_bert.write(ctx_line + ' __sep__ ' + line.strip() + ' __sep__ ' + '0' +'\n')

    def build_dailydialog_bert():
        ctx = '/your/folder/daily_dialog/test/pair_mmi_dialogues_ctx_and_x.txt'
        y_1 = '/your/folder/folder_mirror_nmt/new_mirror_output_nodc10_r234_b10_real.txt'
        y_2 = '/your/folder/folder_mirror_nmt/new_mirror_output_nodc10_r235_b10_real.txt'
        y_3 = '/your/folder/folder_mirror_nmt/new_mirror_output_nodc10_r236_b10_real.txt'
        build_bert_file_step(ctx, y_1, y_1.replace('.txt', '_bert.txt'))
        build_bert_file_step(ctx, y_2, y_2.replace('.txt', '_bert.txt'))
        build_bert_file_step(ctx, y_3, y_3.replace('.txt', '_bert.txt'))
        
        vhred = '/your/folder/VHRED/Output/daily_vhred.txt'
        hred = '/your/folder/VHRED/Output/daily_hred.txt'
        dc_10 = '/your/folder/folder_dc_mmi/new_DC_10_decoding_dailydialog_dev_test_predictions_r1.txt'
        dcmmi_10 = '/your/folder/folder_dc_mmi/new_DCMMI_10_decoding_dailydialog_dev_test_predictions_r1.txt'
        seq2seq = '/your/folder/folder_dc_mmi/new_Seq2Seq_decoding_dailydialog_dev_test_predictions_r2.txt'
        build_bert_file_step(ctx, vhred, vhred.replace('.txt', '_bert.txt'))
        build_bert_file_step(ctx, hred, hred.replace('.txt', '_bert.txt'))
        build_bert_file_step(ctx, dc_10, dc_10.replace('.txt', '_bert.txt'))
        build_bert_file_step(ctx, dcmmi_10, dcmmi_10.replace('.txt', '_bert.txt'))
        build_bert_file_step(ctx, seq2seq, seq2seq.replace('.txt', '_bert.txt'))
    
    def build_movie_bert():
        ctx = '/your/folder/movietriple/test/pair_mmi_dialogues_ctx_and_x.txt'
        y_1 = '/your/folder/folder_mirror_nmt/new_mirror_output_movienodc10_r1_b10_real.txt'

        build_bert_file_step(ctx, y_1, y_1.replace('.txt', '_bert.txt'))
        
        dc_10_movie = '/your/folder/folder_dc_mmi/DC_10_decoding_pair_movie_dev_test_predictions.txt'
        dcmmi_10__movie = '/your/folder/folder_dc_mmi/DCMMI_10_decoding_pair_movie_dev_test_predictions.txt'

        vhred_movie = '/your/folder/VHRED/Output/movie_vhred.txt'
        hred_movie = '/your/folder/VHRED/Output/movie_hred.txt'
        seq2seq = '/your/folder/folder_dc_mmi/new_Seq2Seq_decoding_movie_dev_test_predictions_r2.txt'
        build_bert_file_step(ctx, vhred_movie, vhred_movie.replace('.txt', '_bert.txt'))
        build_bert_file_step(ctx, hred_movie, hred_movie.replace('.txt', '_bert.txt'))
        build_bert_file_step(ctx, dc_10_movie, dc_10_movie.replace('.txt', '_bert.txt'))
        build_bert_file_step(ctx, dcmmi_10__movie, dcmmi_10__movie.replace('.txt', '_bert.txt'))
        build_bert_file_step(ctx, seq2seq, seq2seq.replace('.txt', '_bert.txt'))

    build_dailydialog_bert()
    build_movie_bert()


        

if __name__ == "__main__":




    def build_data_for_seq2seq():
        def construct_two_part_dialog(data, set_):
            path_ctx = '/your/folder/{}/{}/pair_mmi_dialogues_ctx.txt'.format(data, set_)
            path_x = '/your/folder/{}/{}/pair_mmi_dialogues_x.txt'.format(data, set_)
            dest = '/your/folder/{}/{}/pair_mmi_dialogues_ctx_and_x.txt'.format(data, set_)
            def read_files(path):
                with open(path, 'r') as f:
                    data = f.readlines()
                return data
            ctx = read_files(path_ctx)
            x = read_files(path_x)
            assert len(x)==len(ctx)
            with open(dest, 'w') as f:
                for ctx, x in zip(ctx, x):
                    x_new = ctx.strip() + ' __utt__ ' + x.strip()
                    f.write(x_new+ '\n')
        construct_two_part_dialog('daily_dialog', 'train')
        construct_two_part_dialog('daily_dialog', 'validation')
        construct_two_part_dialog('daily_dialog', 'test')

        construct_two_part_dialog('movietriple', 'train')
        construct_two_part_dialog('movietriple', 'validation')
        construct_two_part_dialog('movietriple', 'test')


    def dailydialog_func():
        ctx_path = '/your/folder/daily_dialog/test/pair_mmi_dialogues_ctx.txt'
        x_path = '/your/folder/daily_dialog/test/pair_mmi_dialogues_x.txt'
        y_path = '/your/folder/daily_dialog/test/pair_mmi_dialogues_y.txt'


        builder = build_dialogs(ctx_path, x_path, y_path)

        vhred = '/your/folder/VHRED/Output/daily_vhred.txt'
        hred = '/your/folder/VHRED/Output/daily_hred.txt'

        dc_10 = '/your/folder/folder_dc_mmi/new_DC_10_decoding_dailydialog_dev_test_predictions_r1.txt'
        dcmmi_10 = '/your/folder/folder_dc_mmi/new_DCMMI_10_decoding_dailydialog_dev_test_predictions_r1.txt'
        seq2seq = '/your/folder/folder_dc_mmi/new_Seq2Seq_decoding_dailydialog_dev_test_predictions_r2.txt'
        mmi_50 = '/your/folder/folder_dc_mmi/new_MMI_decoding_dailydialog_dev_test_predictions_r1.txt'
        dcmmi_r2 = '/your/folder/folder_dc_mmi/new_DCMMI_10_decoding_dailydialog_dev_test_predictions_r2.txt'

        # def builder_wrap(idx):
        #     mirror_b10 = '/your/folder/folder_mirror_nmt/new_mirror_output_nodc10_r{}_b10_real.txt'.format(idx)
        #     builder.build_single(mirror_b10, '/your/folder/Mirror-NMT/generated_dialogs/' + 'r{}b10.txt'.format(idx))
        #     builder.build_crowdsourcing_file(mirror_b10, 'mirror', vhred, 'vhred', '/your/folder/Mirror-NMT/generated_dialogs/' + 'r{}b10_vhred.csv'.format(idx))
        #     builder.build_crowdsourcing_file(mirror_b10, 'mirror', hred, 'hred', '/your/folder/Mirror-NMT/generated_dialogs/' + 'r{}b10_hred.csv'.format(idx))
        #     builder.build_crowdsourcing_file(mirror_b10, 'mirror', dc_10, 'dc', '/your/folder/Mirror-NMT/generated_dialogs/' + 'r{}b10_dc.csv'.format(idx))
        #     builder.build_crowdsourcing_file(mirror_b10, 'mirror', dcmmi_10, 'dcmmi', '/your/folder/Mirror-NMT/generated_dialogs/' + 'r{}b10_dcmmi.csv'.format(idx))

        # for exp in range(244, 245):
        #     builder_wrap(exp)
        # mirror_b10_234 = '/your/folder/folder_mirror_nmt/new_mirror_output_nodc10_r234_b10_real.txt'
        # mirror_b10_257 = '/your/folder/folder_mirror_nmt/new_mirror_output_nodc10_r257_b10_real.txt'
        mirror_b10_236 = '/your/folder/folder_mirror_nmt/new_mirror_output_nodc10_r236_b10_real.txt'
        # builder.build_crowdsourcing_file(mirror_b10_236, 'mirror', mirror_b10_257, 'mirrorcat','/your/folder/Mirror-NMT/generated_dialogs/' + 'r236b10_r257.csv')
        # builder.build_crowdsourcing_file(mirror_b10_236, 'mirror', seq2seq, 'seq2seq','/your/folder/Mirror-NMT/generated_dialogs/' + 'r236b10_seq2seq.csv')
        # builder.build_crowdsourcing_file(mirror_b10_236, 'mirror', mmi_50, 'mmi','/your/folder/Mirror-NMT/generated_dialogs/' + 'r236b10_mmi.csv')
        builder.build_crowdsourcing_file(mirror_b10_236, 'mirror', dcmmi_r2, 'dcmmi','/your/folder/Mirror-NMT/generated_dialogs/' + 'r236b10_dcmmir2.csv')

        
    def reddit_func():
        ctx_path = '/your/folder/reddit_dialog/test/pair_mmi_dialogues_ctx.txt'
        x_path = '/your/folder/reddit_dialog/test/pair_mmi_dialogues_x.txt'
        y_path = '/your/folder/reddit_dialog/test/pair_mmi_dialogues_y.txt'
        builder = build_dialogs(ctx_path, x_path, y_path)

        dc_10_reddit = '/your/folder/folder_dc_mmi/new_DC_10_decoding_pair_reddit_dev_test_predictions.txt'
        dcmmi_10_reddit = '/your/folder/folder_dc_mmi/new_DCMMI_10_decoding_pair_reddit_dev_test_predictions.txt'

        vhred_reddit = '/your/folder/VHRED/Output/reddit_vhred.txt'
        hred_reddit = '/your/folder/VHRED/Output/reddit_hred.txt'

        def builder_wrap(idx):
            mirror_b10 = '/your/folder/folder_mirror_nmt/new_mirror_output_redditnodc10_r{}_b10_real.txt'.format(idx)
            builder.build_single(mirror_b10, '/your/folder/Mirror-NMT/generated_dialogs/reddit/reddit_' + 'r{}b10.txt'.format(idx))
            builder.build_crowdsourcing_file(mirror_b10, 'mirror', vhred_reddit, 'vhred', '/your/folder/Mirror-NMT/generated_dialogs/reddit/reddit_' + 'r{}b10_vhred.csv'.format(idx))
            builder.build_crowdsourcing_file(mirror_b10, 'mirror', hred_reddit, 'hred', '/your/folder/Mirror-NMT/generated_dialogs/reddit/reddit_' + 'r{}b10_hred.csv'.format(idx))
            builder.build_crowdsourcing_file(mirror_b10, 'mirror', dc_10_reddit, 'dc', '/your/folder/Mirror-NMT/generated_dialogs/reddit/reddit_' + 'r{}b10_dc.csv'.format(idx))
            builder.build_crowdsourcing_file(mirror_b10, 'mirror', dcmmi_10_reddit, 'dcmmi', '/your/folder/Mirror-NMT/generated_dialogs/reddit/reddit_' + 'r{}b10_dcmmi.csv'.format(idx))

        for idx in range(1,11):
            builder_wrap(idx)

    def movie_func():
        ctx_path = '/your/folder/movietriple/test/pair_mmi_dialogues_ctx.txt'
        x_path = '/your/folder/movietriple/test/pair_mmi_dialogues_x.txt'
        y_path = '/your/folder/movietriple/test/pair_mmi_dialogues_y.txt'
        builder = build_dialogs(ctx_path, x_path, y_path)

        dc_10_movie = '/your/folder/folder_dc_mmi/DC_10_decoding_pair_movie_dev_test_predictions.txt'
        dcmmi_10__movie = '/your/folder/folder_dc_mmi/DCMMI_10_decoding_pair_movie_dev_test_predictions.txt'

        vhred_movie = '/your/folder/VHRED/Output/movie_vhred.txt'
        hred_movie = '/your/folder/VHRED/Output/movie_hred.txt'
        seq2seq = '/your/folder/folder_dc_mmi/new_Seq2Seq_decoding_movie_dev_test_predictions_r2.txt'
        # mmi_50 = '/your/folder/folder_dc_mmi/MMI_10_decoding_pair_movie_dev_test_predictions_b50.txt'
        dcmmi_r2_movie =  '/your/folder/folder_dc_mmi/DCMMI_10_decoding_pair_movie_dev_test_predictions_r2.txt'

        # def builder_wrap(idx):
            # mirror_b10 = '/your/folder/folder_mirror_nmt/new_mirror_output_movienodc10_r{}_b10_real.txt'.format(idx)
            # builder.build_single(mirror_b10, '/your/folder/Mirror-NMT/generated_dialogs/movie/movie_' + 'r{}b10.txt'.format(idx))
            # builder.build_crowdsourcing_file(mirror_b10, 'mirror', vhred_movie, 'vhred', '/your/folder/Mirror-NMT/generated_dialogs/movie/movie_' + 'r{}b10_vhred.csv'.format(idx))
            # builder.build_crowdsourcing_file(mirror_b10, 'mirror', hred_movie, 'hred', '/your/folder/Mirror-NMT/generated_dialogs/movie/movie_' + 'r{}b10_hred.csv'.format(idx))
            # builder.build_crowdsourcing_file(mirror_b10, 'mirror', dc_10_movie, 'dc', '/your/folder/Mirror-NMT/generated_dialogs/movie/movie_' + 'r{}b10_dc.csv'.format(idx))
            # builder.build_crowdsourcing_file(mirror_b10, 'mirror', dcmmi_10__movie, 'dcmmi', '/your/folder/Mirror-NMT/generated_dialogs/movie/movie_' + 'r{}b10_dcmmi.csv'.format(idx))
            # builder.build_crowdsourcing_file(mirror_b10, 'mirror', seq2seq, 'seq2seq', '/your/folder/Mirror-NMT/generated_dialogs/movie/movie_' + 'r{}b10_dcmmi.csv'.format(idx))

        mirror_b10 = '/your/folder/folder_mirror_nmt/new_mirror_output_movienodc10_r1_b10_real.txt'
        # builder.build_crowdsourcing_file(mirror_b10, 'mirror', seq2seq, 'seq2seq', '/your/folder/Mirror-NMT/generated_dialogs/movie/movie_r1b10_seq2seq.csv')
        # builder.build_crowdsourcing_file(mirror_b10, 'mirror', mmi_50, 'mmi', '/your/folder/Mirror-NMT/generated_dialogs/movie/movie_r1b10_mmi.csv')
        builder.build_crowdsourcing_file(mirror_b10, 'mirror', dcmmi_r2_movie, 'dcmmi', '/your/folder/Mirror-NMT/generated_dialogs/movie/movie_r1b10_dcmmir2.csv')


        # for idx in [1]:
            # builder_wrap(idx)
        
    def build_all_dialogues():
        ctx_path = '/your/folder/daily_dialog/test/pair_mmi_dialogues_ctx.txt'
        x_path = '/your/folder/daily_dialog/test/pair_mmi_dialogues_x.txt'
        y_path = '/your/folder/daily_dialog/test/pair_mmi_dialogues_y.txt'
        builder = build_dialogs(ctx_path, x_path, y_path)
        source_dict = {
            'Seq2Seq': '/your/folder/folder_dc_mmi/new_Seq2Seq_decoding_dailydialog_dev_test_predictions_r2.txt',
            'HRED': '/your/folder/VHRED/Output/daily_hred.txt',
            'VHRED': '/your/folder/VHRED/Output/daily_vhred.txt',
            'MMI': '/your/folder/folder_dc_mmi/new_MMI_decoding_dailydialog_dev_test_predictions_r1.txt',
            'DC': '/your/folder/folder_dc_mmi/new_DC_10_decoding_dailydialog_dev_test_predictions_r1.txt',
            'DCMMI': '/your/folder/folder_dc_mmi/new_DCMMI_10_decoding_dailydialog_dev_test_predictions_r2.txt',
            'MIRROR': '/your/folder/folder_mirror_nmt/new_mirror_output_nodc10_r236_b10_real.txt'
        }
        dest = '/your/folder/Mirror-NMT/generated_dialogs/all_daily.txt'
        builder.build_multiple(source_dict, dest)
    #  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        ctx_path = '/your/folder/movietriple/test/pair_mmi_dialogues_ctx.txt'
        x_path = '/your/folder/movietriple/test/pair_mmi_dialogues_x.txt'
        y_path = '/your/folder/movietriple/test/pair_mmi_dialogues_y.txt'
        builder = build_dialogs(ctx_path, x_path, y_path)

        source_dict = {
            'Seq2Seq': '/your/folder/folder_dc_mmi/new_Seq2Seq_decoding_movie_dev_test_predictions_r2.txt',
            'HRED': '/your/folder/VHRED/Output/movie_hred.txt',
            'VHRED': '/your/folder/VHRED/Output/movie_vhred.txt',
            'MMI': '/your/folder/folder_dc_mmi/MMI_10_decoding_pair_movie_dev_test_predictions_b50.txt',
            'DC': '/your/folder/folder_dc_mmi/DC_10_decoding_pair_movie_dev_test_predictions.txt',
            'DCMMI': '/your/folder/folder_dc_mmi/DCMMI_10_decoding_pair_movie_dev_test_predictions_r2.txt',
            'MIRROR': '/your/folder/folder_mirror_nmt/new_mirror_output_movienodc10_r1_b10_real.txt',
        }
        dest = '/your/folder/Mirror-NMT/generated_dialogs/all_movie.txt'
        builder.build_multiple(source_dict, dest)

    # build_all_dialogues()
    # reddit_func()
    # dailydialog_func()
    movie_func()
    # build_data_for_seq2seq()
    # build_bert_files()

