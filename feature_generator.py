import numpy as np
import pandas as pd


class Feature:
    def __init__(self, feature_str, metric, feature_len):
        self.feature_str = feature_str
        self.metric = metric
        self.feature_len = feature_len

    def calc(self, df):
        calc_string = self.feature_str.split(' ')

        result = df[calc_string[0]]
        operator = None

        for i, item in enumerate(calc_string):
            if item in ['*', '/', '%', '+', '-']:
                operator = item
            else:

                if operator == '*':
                    result = result * df[item]
                elif operator == '/':
                    result = result / df[item]
                elif operator == '-':
                    result = result - df[item]
                elif operator == '+':
                    result = result + df[item]
                elif operator == '%':
                    result = result % df[item]

                operator = None

        return result

    def add(self, feature_name, operator, metric):
        self.feature_str += (' ' + operator)
        self.feature_str += (' ' + feature_name)

        self.metric = metric
        self.feature_len += 1

    def get_last_feature(self):
        calc_string = self.feature_str.split(' ')
        return calc_string[-1] if len(calc_string[-1]) > 1 else calc_string[-2]


    def __lt__(self, other):
        return abs(self.metric) < abs(other.metric)

    def __gt__(self, other):
        return abs(self.metric) > abs(other.metric)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class Features_Generator_24V_1Kw:
    '''
    Working by using correlation
    
    Algorithm 1:
    1) Select i'th random feature, mark as used
    2) Select uncorrelated feature, mark as used
    3) Transform first feature with second
    Repeat
    '''

    def __init__(self, df, target_var, features, max_feats_combination=5):
        self.df = df
        self.target_var = target_var
        self.max_feats_combination = max_feats_combination

        print('Finding correlation matrix')
        self.corr = self.df[features].corr()
        print('Done!')

        self.features = features
        np.random.shuffle(self.features)
        self.unused_features = features
        self.generated_features = []

    def calc_correlation(self, feature):
        corr = self.df[self.target_var].corr(feature)

        return corr

    def calc_correlation_by_name(self, feature_name):
        corr = self.df[self.target_var].corr(self.df[feature_name])

        return corr

    def cross_feat_correlation(self, feature, test_feature):
        corr = self.df[feature].corr(test_feature)

        return corr

    def gen_alg1(self, feature):
        if feature.feature_len < self.max_feats_combination and self.corr.shape[0] > 0:

            current_feat = feature.calc(self.df)

            correlation_array = [self.cross_feat_correlation(temp_feat, current_feat) for temp_feat in
                                 self.unused_features]
            max_uncorrelated = find_nearest(correlation_array, 0)

            if abs(correlation_array[max_uncorrelated]) > 0.2:
                return feature

            next_feature = self.corr[feature.get_last_feature()].index[max_uncorrelated]
            # print(self.unused_features, next_feature)

            # Remove feature from unused
            self.corr.drop(index=next_feature, inplace=True)
            self.unused_features.remove(next_feature)

            results = []

            results.append(self.calc_correlation(current_feat * self.df[next_feature]))
            results.append(self.calc_correlation(current_feat / self.df[next_feature]))
            results.append(self.calc_correlation(current_feat - self.df[next_feature]))
            results.append(self.calc_correlation(current_feat + self.df[next_feature]))
            results.append(self.calc_correlation(current_feat % self.df[next_feature]))

            # print(results, current_feat)
            results = np.nan_to_num(np.array(results), 0)

            best = max(results.min(), results.max(), key=abs)
            best_idx = np.where(results == best)[0][0]
            #
            #  print(best, results)

            if best_idx == 0:
                feature.add(next_feature, '*', best)
            if best_idx == 1:
                feature.add(next_feature, '/', best)
            if best_idx == 2:
                feature.add(next_feature, '-', best)
            if best_idx == 3:
                feature.add(next_feature, '+', best)
            if best_idx == 4:
                feature.add(next_feature, '%', best)

            feature = self.gen_alg1(feature)

        return feature



    def gen_alg2(self, feature):
        if feature.feature_len < self.max_feats_combination and len(self.unused_features) > 0: #self.corr.shape[0]

            # # Finding next most correlated feature with target
            # correlation_array = [self.calc_correlation_by_name(temp_feat) for temp_feat in
            #                      self.unused_features]
            # correlation_array = np.nan_to_num(np.array(correlation_array), 0)
            #
            # best = max(correlation_array.min(), correlation_array.max(), key=abs)
            # best_idx = np.where(correlation_array == best)[0][0]
            # next_feature_name = self.corr.index[best_idx]
            #
            # # Remove feature from unused
            # self.corr.drop(index=next_feature_name, inplace=True)
            # self.unused_features.remove(next_feature_name)

            #feats_name_array = []
            correlation_array = []
            operation_array = []

            for temp_feature in self.unused_features:

                # Finding best operation
                results = []
                current_feat = feature.calc(self.df)

                results.append(self.calc_correlation(current_feat * self.df[temp_feature]))
                results.append(self.calc_correlation(current_feat / self.df[temp_feature]))
                results.append(self.calc_correlation(current_feat - self.df[temp_feature]))
                results.append(self.calc_correlation(current_feat + self.df[temp_feature]))
                results.append(self.calc_correlation(current_feat % self.df[temp_feature]))

                # print(results, current_feat)
                results = np.nan_to_num(np.array(results), 0)

                best = max(results.min(), results.max(), key=abs)
                correlation_array.append(best)

                best_idx = np.where(results == best)[0][0]


                if best_idx == 0:
                    operation_array.append('*')
                if best_idx == 1:
                    operation_array.append('/')
                if best_idx == 2:
                    operation_array.append('-')
                if best_idx == 3:
                    operation_array.append('+')
                if best_idx == 4:
                    operation_array.append('%')






            correlation_array = np.nan_to_num(np.array(correlation_array), 0)

            best = max(correlation_array.min(), correlation_array.max(), key=abs)
            if abs(best) > abs(feature.metric):
                best_idx = np.where(correlation_array == best)[0][0]
                next_feature_name = self.unused_features[best_idx]
                feature.add(next_feature_name, operation_array[best_idx], correlation_array[best_idx])

                # Remove feature from unused
                #self.corr.drop(index=next_feature_name, inplace=True)
                self.unused_features.remove(next_feature_name)
                feature = self.gen_alg2(feature)

        return feature



    def start(self):
        i = 0
        for feature_name in self.corr.columns:
        #while self.corr.shape[0] > 0:
            print(f'Generating {i}th pack....')

            # # Finding most correlated feature with target
            # correlation_array = [self.calc_correlation_by_name(temp_feat) for temp_feat in
            #                      self.unused_features]
            # correlation_array = np.nan_to_num(np.array(correlation_array), 0)
            #
            # best = max(correlation_array.min(), correlation_array.max(), key=abs)
            # best_idx = np.where(correlation_array == best)[0][0]
            # feature_name = self.corr.index[best_idx]

            # Remove feature from unused
            # self.corr.drop(index=feature_name, inplace=True)
            # self.unused_features.remove(feature_name)

            temp_feat = Feature(feature_name, self.calc_correlation_by_name(feature_name), 1)
            print(f'Metric: {temp_feat.metric}')
            self.generated_features.append(self.gen_alg2(temp_feat))
            i += 1

        print(f'All feats were utilized...')
        return self.generated_features

    def get_df(self, generated_features=None, df=None):
        if generated_features is None:
            generated_features = self.generated_features
            
        if df is None:
            df = self.df

        result_df = pd.DataFrame()
        result_df['resp'] = df['resp']
        result_df['resp_1'] = df['resp_1']
        result_df['resp_2'] = df['resp_2']
        result_df['resp_3'] = df['resp_3']
        result_df['resp_4'] = df['resp_4']
        result_df['date'] = df['date']
        result_df['weight'] = df['weight']
        result_df['ts_id'] = df['ts_id']
        for i, gened_feature in enumerate(generated_features):
            result_df[f'gened_feat_{i}'] = gened_feature.calc(df)

        return result_df


def load_data():
    # Load 50 rows to understand types of cols
    train_csv = pd.read_csv('train_small.csv', nrows=50)
    # Finding float cols
    float_cols = [c for c in train_csv if train_csv[c].dtype == "float64"]
    # Float64 to Float16
    types = {c: 'float16' for c in float_cols}
    # Loading all rows
    train_csv = pd.read_csv('train_small.csv', engine='c', dtype=types)

    return train_csv

if __name__ == "__main__":
   train_csv = load_data()

   EBAT = Features_Generator_24V_1Kw(train_csv, 'resp', ["feature_%d" % i for i in range(130)], 3)

   rez = EBAT.start()