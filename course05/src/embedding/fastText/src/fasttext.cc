/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "fasttext.h"

#include <math.h>

#include <iostream>
#include <iomanip>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>

namespace fasttext
{

void FastText::getVector(Vector &vec, const std::string &word)
{
  const std::vector<int32_t> &ngrams = dict_->getNgrams(word);
  vec.zero();
  for (auto it = ngrams.begin(); it != ngrams.end(); ++it)
  {
    vec.addRow(*input_, *it);
  }
  if (ngrams.size() > 0)
  {
    vec.mul(1.0 / ngrams.size());
  }
}

void FastText::saveVectors()
{
  std::ofstream ofs(args_->output + ".vec");
  if (!ofs.is_open())
  {
    std::cout << "Error opening file for saving vectors." << std::endl;
    exit(EXIT_FAILURE);
  }
  ofs << dict_->nwords() << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords(); i++)
  {
    std::string word = dict_->getWord(i);
    getVector(vec, word);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void FastText::saveOutput()
{
  std::ofstream ofs(args_->output + ".output");
  if (!ofs.is_open())
  {
    std::cout << "Error opening file for saving vectors." << std::endl;
    exit(EXIT_FAILURE);
  }
  ofs << dict_->nwords() << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords(); i++)
  {
    std::string word = dict_->getWord(i);
    vec.zero();
    vec.addRow(*output_, i);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void FastText::saveModel()
{
  std::ofstream ofs(args_->output + ".bin", std::ofstream::binary);
  if (!ofs.is_open())
  {
    std::cerr << "Model file cannot be opened for saving!" << std::endl;
    exit(EXIT_FAILURE);
  }
  args_->save(ofs);
  dict_->save(ofs);
  input_->save(ofs);
  output_->save(ofs);
  ofs.close();
}

void FastText::loadModel(const std::string &filename)
{
  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open())
  {
    std::cerr << "Model file cannot be opened for loading!" << std::endl;
    exit(EXIT_FAILURE);
  }
  loadModel(ifs);
  ifs.close();
}

void FastText::loadModel(std::istream &in)
{
  args_ = std::make_shared<Args>();
  dict_ = std::make_shared<Dictionary>(args_);
  input_ = std::make_shared<Matrix>();
  output_ = std::make_shared<Matrix>();
  args_->load(in);
  dict_->load(in);
  input_->load(in);
  output_->load(in);
  model_ = std::make_shared<Model>(input_, output_, args_, 0);
  if (args_->model == model_name::sup)
  {
    model_->setTargetCounts(dict_->getCounts(entry_type::label));
  }
  else
  {
    model_->setTargetCounts(dict_->getCounts(entry_type::word));
  }
}

void FastText::printInfo(real progress, real loss)
{
  real t = real(clock() - start) / CLOCKS_PER_SEC;
  real wst = real(tokenCount) / t;
  real lr = args_->lr * (1.0 - progress);
  int eta = int(t / progress * (1 - progress) / args_->thread);
  int etah = eta / 3600;
  int etam = (eta - etah * 3600) / 60;
  std::cout << std::fixed;
  std::cout << "\rProgress: " << std::setprecision(1) << 100 * progress << "%";
  std::cout << "  words/sec/thread: " << std::setprecision(0) << wst;
  std::cout << "  lr: " << std::setprecision(6) << lr;
  std::cout << "  loss: " << std::setprecision(6) << loss;
  std::cout << "  eta: " << etah << "h" << etam << "m ";
  std::cout << std::flush;
}

void FastText::supervised(Model &model, real lr,
                          const std::vector<int32_t> &line,
                          const std::vector<int32_t> &labels)
{
  if (labels.size() == 0 || line.size() == 0)
    return;
  // 这里随机的选取了多标签分类中的一个标签作为label来更新模型
  // 所以fastText并不适合进行多标签问题的分类
  std::uniform_int_distribution<> uniform(0, labels.size() - 1);
  int32_t i = uniform(model.rng);
  model.update(line, labels[i], lr);
}

void FastText::cbow(Model &model, real lr,
                    const std::vector<int32_t> &line)
{
  std::vector<int32_t> bow;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  // 窗口在句子上从左到右滑动，所以每滑动一个单词模型更新一次
  for (int32_t w = 0; w < line.size(); w++)
  {
    // 上下文窗口是随机的 [1, windowsize]
    int32_t boundary = uniform(model.rng);
    bow.clear();
    // bow装的就是上下文单词
    for (int32_t c = -boundary; c <= boundary; c++)
    {
      if (c != 0 && w + c >= 0 && w + c < line.size())
      {
        const std::vector<int32_t> &ngrams = dict_->getNgrams(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    model.update(bow, line[w], lr);
  }
}

void FastText::skipgram(Model &model, real lr,
                        const std::vector<int32_t> &line)
{
  std::uniform_int_distribution<> uniform(1, args_->ws);
  // 目标词换成了词的Ngrams，然后每构造一个[target,contextword]的样本就进行一次更新
  for (int32_t w = 0; w < line.size(); w++)
  {
    int32_t boundary = uniform(model.rng);
    const std::vector<int32_t> &ngrams = dict_->getNgrams(line[w]);
    for (int32_t c = -boundary; c <= boundary; c++)
    {
      if (c != 0 && w + c >= 0 && w + c < line.size())
      {
        model.update(ngrams, line[w + c], lr);
      }
    }
  }
}

void FastText::test(std::istream &in, int32_t k)
{
  int32_t nexamples = 0, nlabels = 0;
  double precision = 0.0;
  std::vector<int32_t> line, labels;

  while (in.peek() != EOF)
  {
    dict_->getLine(in, line, labels, model_->rng);
    dict_->addNgrams(line, args_->wordNgrams);
    if (labels.size() > 0 && line.size() > 0)
    {
      std::vector<std::pair<real, int32_t>> modelPredictions;
      model_->predict(line, k, modelPredictions);
      for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++)
      {
        if (std::find(labels.begin(), labels.end(), it->second) != labels.end())
        {
          precision += 1.0;
        }
      }
      nexamples++;
      nlabels += labels.size();
    }
  }
  std::cout << std::setprecision(3);
  std::cout << "P@" << k << ": " << precision / (k * nexamples) << std::endl;
  std::cout << "R@" << k << ": " << precision / nlabels << std::endl;
  std::cout << "Number of examples: " << nexamples << std::endl;
}

void FastText::predict(std::istream &in, int32_t k,
                       std::vector<std::pair<real, std::string>> &predictions) const
{
  std::vector<int32_t> words, labels;
  dict_->getLine(in, words, labels, model_->rng);
  dict_->addNgrams(words, args_->wordNgrams);
  if (words.empty())
    return;
  Vector hidden(args_->dim);
  Vector output(dict_->nlabels());
  std::vector<std::pair<real, int32_t>> modelPredictions;
  model_->predict(words, k, modelPredictions, hidden, output);
  predictions.clear();
  for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++)
  {
    predictions.push_back(std::make_pair(it->first, dict_->getLabel(it->second)));
  }
}

void FastText::predict(std::istream &in, int32_t k, bool print_prob)
{
  std::vector<std::pair<real, std::string>> predictions;
  while (in.peek() != EOF)
  {
    predict(in, k, predictions);
    if (predictions.empty())
    {
      std::cout << "n/a" << std::endl;
      continue;
    }
    for (auto it = predictions.cbegin(); it != predictions.cend(); it++)
    {
      if (it != predictions.cbegin())
      {
        std::cout << ' ';
      }
      std::cout << it->second;
      if (print_prob)
      {
        std::cout << ' ' << exp(it->first);
      }
    }
    std::cout << std::endl;
  }
}

void FastText::wordVectors()
{
  std::string word;
  Vector vec(args_->dim);
  while (std::cin >> word)
  {
    getVector(vec, word);
    std::cout << word << " " << vec << std::endl;
  }
}

void FastText::ngramVectors(std::string word)
{
  std::vector<int32_t> ngrams;
  std::vector<std::string> substrings;
  Vector vec(args_->dim);
  dict_->getNgrams(word, ngrams, substrings);
  for (int32_t i = 0; i < ngrams.size(); i++)
  {
    vec.zero();
    if (ngrams[i] >= 0)
    {
      vec.addRow(*input_, ngrams[i]);
    }
    std::cout << substrings[i] << " " << vec << std::endl;
  }
}

void FastText::textVectors()
{
  std::vector<int32_t> line, labels;
  Vector vec(args_->dim);
  while (std::cin.peek() != EOF)
  {
    dict_->getLine(std::cin, line, labels, model_->rng);
    dict_->addNgrams(line, args_->wordNgrams);
    vec.zero();
    for (auto it = line.cbegin(); it != line.cend(); ++it)
    {
      vec.addRow(*input_, *it);
    }
    if (!line.empty())
    {
      vec.mul(1.0 / line.size());
    }
    std::cout << vec << std::endl;
  }
}

void FastText::printVectors()
{
  if (args_->model == model_name::sup)
  {
    textVectors();
  }
  else
  {
    wordVectors();
  }
}

/**
 * 训练的线程函数，主要的训练过程。
 * 调用了不同任务的参数更新策略(所谓训练，就是参数的更新过程，这里将训练抽象成了参数的更新过程)
 * 多线程的训练当中每个线程并没有加锁，所以会给参数的更新带来了一些噪音，但是不会影响最终的结果
 * 模型的更新策略实际发生在 supervised、cbow、skipgram三个函数当中，这3个函数都同时调用同一个
 * model.update() 函数来更新参数
 */
void FastText::trainThread(int32_t threadId)
{
  std::ifstream ifs(args_->input);
  // 将训练文件均分成thread份，然后通过threadId寻找到相应的部分
  // 将文件指针移动到那个相应的地方
  utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);

  Model model(input_, output_, args_, threadId);
  if (args_->model == model_name::sup)
  {
    model.setTargetCounts(dict_->getCounts(entry_type::label));
  }
  else
  {
    model.setTargetCounts(dict_->getCounts(entry_type::word));
  }
  // 训练文件中的token总数目
  const int64_t ntokens = dict_->ntokens();
  // 当前线程处理的token总数目
  int64_t localTokenCount = 0;
  std::vector<int32_t> line, labels;
  // tokenCount为所有线程处理完毕之后的token总数目
  while (tokenCount < args_->epoch * ntokens)
  {
    // 当前处理进度，用于显示处理过程
    real progress = real(tokenCount) / (args_->epoch * ntokens);
    // 学习率根据处理过程线性减小
    real lr = args_->lr * (1.0 - progress);
    localTokenCount += dict_->getLine(ifs, line, labels, model.rng);
    // 根据命令行的配置不同，模型参数的更新策略也是不同的，分为以下3个不同的策略：
    // 1. 有监督学习（分类任务）
    // 2. word2vec （CBOW)
    // 3. word2vec （SKIPGRAM)
    if (args_->model == model_name::sup)
    {
      dict_->addNgrams(line, args_->wordNgrams);
      supervised(model, lr, line, labels);
    }
    else if (args_->model == model_name::cbow)
    {
      cbow(model, lr, line);
    }
    else if (args_->model == model_name::sg)
    {
      skipgram(model, lr, line);
    }
    // lrUpdateRate(default 100) 每个线程学习率的变化率 
    // localTokenCount 影响 tokenCount 影响 progress 影响 lr
    if (localTokenCount > args_->lrUpdateRate)
    {
      tokenCount += localTokenCount;
      localTokenCount = 0;
      // 如果是第一个线程的话，那么第一个线程负责打印消息，其它线程不用打印
      if (threadId == 0 && args_->verbose > 1)
      {
        printInfo(progress, model.getLoss());
      }
    }
  }
  // 训练完毕的时候，输出总计信息
  if (threadId == 0 && args_->verbose > 0)
  {
    printInfo(1.0, model.getLoss());
    std::cout << std::endl;
  }
  ifs.close();
}

void FastText::loadVectors(std::string filename)
{
  std::ifstream in(filename);
  std::vector<std::string> words;
  std::shared_ptr<Matrix> mat; // temp. matrix for pretrained vectors
  int64_t n, dim;
  if (!in.is_open())
  {
    std::cerr << "Pretrained vectors file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  in >> n >> dim;
  if (dim != args_->dim)
  {
    std::cerr << "Dimension of pretrained vectors does not match -dim option"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  mat = std::make_shared<Matrix>(n, dim);
  for (size_t i = 0; i < n; i++)
  {
    std::string word;
    in >> word;
    words.push_back(word);
    dict_->add(word);
    for (size_t j = 0; j < dim; j++)
    {
      in >> mat->data_[i * dim + j];
    }
  }
  in.close();

  dict_->threshold(1, 0);
  input_ = std::make_shared<Matrix>(dict_->nwords() + args_->bucket, args_->dim);
  input_->uniform(1.0 / args_->dim);

  for (size_t i = 0; i < n; i++)
  {
    int32_t idx = dict_->getId(words[i]);
    if (idx < 0 || idx >= dict_->nwords())
      continue;
    for (size_t j = 0; j < dim; j++)
    {
      input_->data_[idx * dim + j] = mat->data_[i * dim + j];
    }
  }
}

void FastText::train(std::shared_ptr<Args> args)
{
  args_ = args;
  dict_ = std::make_shared<Dictionary>(args_);
  if (args_->input == "-")
  {
    // manage expectations
    std::cerr << "Cannot use stdin for training!" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::ifstream ifs(args_->input);
  if (!ifs.is_open())
  {
    std::cerr << "Input file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  // 根据输入语料库文件初始化词典
  dict_->readFromFile(ifs);
  ifs.close();
  // 如果是在命令行中指定了有已经预先训练好的词向量，
  // 那么加载这些词向量，否则自己申请内存从头开始训练
  if (args_->pretrainedVectors.size() != 0)
  {
    loadVectors(args_->pretrainedVectors);
  }
  else
  {
    // 初始化输入层，对于普通的word2vec，输入层就是一个词向量的查找表，大小为[nwords,dim]
    // fastText又添加了n-gram，所以输出矩阵的大小为 [nwords+ngram, dim]
    // 在代码中，所有n-gram 都被hash到固定数目的bucket中，所以输出矩阵的大小为 [nwords+bucket, dim]
    input_ = std::make_shared<Matrix>(dict_->nwords() + args_->bucket, args_->dim);
    input_->uniform(1.0 / args_->dim);
  }
  // 如果是监督学习分类的话，输出层对应的是label的个数，如果训练词向量的话，输出层对应词典的大小
  if (args_->model == model_name::sup)
  {
    output_ = std::make_shared<Matrix>(dict_->nlabels(), args_->dim);
  }
  else
  {
    output_ = std::make_shared<Matrix>(dict_->nwords(), args_->dim);
  }
  output_->zero();

  start = clock();
  tokenCount = 0;
  std::vector<std::thread> threads;
  for (int32_t i = 0; i < args_->thread; i++)
  {
    threads.push_back(std::thread([=]() { trainThread(i); }));
  }
  for (auto it = threads.begin(); it != threads.end(); ++it)
  {
    it->join();
  }
  model_ = std::make_shared<Model>(input_, output_, args_, 0);

  saveModel();
  if (args_->model != model_name::sup)
  {
    saveVectors();
    if (args_->saveOutput > 0)
    {
      saveOutput();
    }
  }
}
}
