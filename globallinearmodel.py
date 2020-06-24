#预处理文本类
class Text_preprocess:
    def __init__(self, file):
        self.sentences = []
        self.taglists = []
        self.tag = set()
        self.tag2index = dict()
        self.preprocess(file)
    def preprocess(self, filename):
        with open(filename, 'r', encoding='utf-8') as f1:
            lines = [line.strip().split('\t') for line in f1.readlines()]
        sent = []
        taglist = []
        for line in lines:
            if line==['']:
                self.sentences.append(sent)
                self.taglists.append(taglist)
                sent = []
                taglist = []
            else:
                sent.append(line[1])
                taglist.append(line[3])
                self.tag.add(line[3])
        i = 0
        for t in self.tag:
            self.tag2index[t] = i
            i += 1

class Globalmodel:
    #初始化特征集合，索引，权重矩阵
    def __init__(self, text: Text_preprocess):
        self.featureset = set()
        self.feature2index = dict()
        self.preoperator(text)
        self.weight = [[0. for _ in range(len(self.featureset))] for k in range(len(text.tag))]
        self.tag2index = text.tag2index
        self.tagset = text.tag
    #建立特征集与特征索引
    def preoperator(self, text:Text_preprocess):
        for i in range(len(text.sentences)):
            for pos in range(len(text.sentences[i])):
                if pos == 0:
                    feature = self.exa_feature(text.sentences[i], pos, '*')
                else:
                    feature = self.exa_feature(text.sentences[i], pos, text.taglists[i][pos-1])
                for f in feature:
                    self.featureset.add(f)
        n = 0
        for f in self.featureset:
            self.feature2index[f] = n
            n += 1
    #特征提取函数
    def exa_feature(self, sent, pos, pretag):
        f = []
        word = sent[pos]
        if pos == len(sent)-1:
            nextword = '$$'
        else:
            nextword = sent[pos+1]
        if pos == 0:
            pre = '**'
        else:
            pre = sent[pos-1]
        f.append('01:'+pretag)
        f.append('02:'+word)
        f.append('03:'+pre)
        f.append('04:'+nextword)
        f.append('05:'+word+pre[-1])
        f.append('06:'+word+nextword[0])
        f.append('07:'+word[0])
        f.append('08:'+word[-1])
        for i in range(1, len(word)-1):
            f.append('09:'+word[i])
            f.append('10:'+word[0]+word[i])
            f.append('11:'+word[-1]+word[i])
        if len(word)==1:
            f.append('12:'+word+pre[-1]+nextword[0])
        for i in range(0, len(word)-2):
            if word[i]==word[i+1]:
                f.append('13:'+word[i]+'consecutive')
        if len(word)>=4:
            for k in range(4):
                f.append('14:'+word[:k+1])
                f.append('15:'+word[-k-1::])
        return f
    #计算单个句子一个位置词性为tag的得分
    def computer_score(self, sent, pos, pretag, tag):
        feature = self.exa_feature(sent, pos, pretag)
        feature = [self.feature2index[f] for f in feature if f in self.featureset]
        tagindex = self.tag2index[tag]
        score = 0
        for f in feature:
            score += self.weight[tagindex][f]
        return score, feature
    #维特比动态规划预测序列
    def viterbi(self, sent):
        curtag = '*'
        fmaxtrix = []
        predicttags = []
        for pos in range(len(sent)):
            pretag = curtag
            maxnum = -1e10
            for t in self.tagset:
                score, feature = self.computer_score(sent, pos, pretag, t)
                if score>maxnum:
                    maxnum = score
                    curtag = t
                    maxfeature = feature
            predicttags.append(curtag)
            fmaxtrix.append(maxfeature)
        return predicttags, fmaxtrix
    #Online Training训练weight
    def Onlinetraining(self, text:Text_preprocess, num):
        for iterator in range(num):
            for i in range(len(text.sentences)):
                predicttags, predictf = self.viterbi(text.sentences[i])
                if predicttags!=text.taglists[i]:
                    for j in range(len(predictf)):
                        for f in predictf[j]:
                            self.weight[self.tag2index[predicttags[j]]][f] -= 1
                        if j==0:
                            _, fcorret = self.computer_score(text.sentences[i], j, '*', text.taglists[i][j])
                        else:
                            _, fcorret = self.computer_score(text.sentences[i], j, text.taglists[i][j-1], text.taglists[i][j])
                        for f in fcorret:
                            self.weight[self.tag2index[text.taglists[i][j]]][f] += 1
            precision = self.evaluate(text)
            print('第%d次迭代，精度为：'%iterator, precision)
    def evaluate(self, text:Text_preprocess):
        count = 0
        right = 0
        for i in range(len(text.sentences)):
            predicttags, predictf = self.viterbi(text.sentences[i])
            for j in range(len(predicttags)):
                count += 1
                if predicttags[j]==text.taglists[i][j]:
                    right += 1
        return right/count
if __name__ == '__main__':
    train = Text_preprocess('data/train.conll')
    dev = Text_preprocess('data/dev.conll')
    model = Globalmodel(train)
    model.Onlinetraining(train, 10)
    print('测试集上的精度为：', model.evaluate(dev))








