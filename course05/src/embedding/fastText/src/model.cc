/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "model.h"

#include <assert.h>

#include <algorithm>

namespace fasttext
{

Model::Model(std::shared_ptr<Matrix> wi,
             std::shared_ptr<Matrix> wo,
             std::shared_ptr<Args> args,
             int32_t seed)
    : hidden_(args->dim), output_(wo->m_), grad_(args->dim), rng(seed)
{
  wi_ = wi;
  wo_ = wo;
  args_ = args;
  isz_ = wi->m_;
  osz_ = wo->m_;
  hsz_ = args->dim;
  negpos = 0;
  loss_ = 0.0;
  nexamples_ = 1;
  initSigmoid();
  initLog();
}

Model::~Model()
{
  delete[] t_sigmoid;
  delete[] t_log;
}

/**
 * Model将hierarchical softmax 和Negative sampling统一抽象成多个二元logistic regression 的连乘
 */
real Model::binaryLogistic(int32_t target, bool label, real lr)
{
  // 将hidden_和参数矩阵的第target 行做内积，计算sigmoid
  real score = sigmoid(wo_->dotRow(hidden_, target));
  real alpha = lr * (real(label) - score);
  grad_.addRow(*wo_, target, alpha);
  // 每一个二分类器的参数是及时更新的，词向量wi_是在走过所有的二分类器之后才更新的
  wo_->addRow(hidden_, target, alpha);
  // 在负采样的时候，正负样本返回的loss是不一样的
  if (label)
  {
    return -log(score);
  }
  else
  {
    return -log(1.0 - score);
  }
}
/**
 * 这里每一个更新方法都是在构造输出层的结构
 * 然后调用一连串的二分类器，将误差累计
 */ 
real Model::negativeSampling(int32_t target, real lr)
{
  real loss = 0.0;
  grad_.zero();
  // 构造一个正样本，然后构造neg个负样本，label就是样本的label
  for (int32_t n = 0; n <= args_->neg; n++)
  {
    if (n == 0)
    {
      loss += binaryLogistic(target, true, lr);
    }
    else
    {
      loss += binaryLogistic(getNegative(target), false, lr);
    }
  }
  return loss;
}
/**
 * 这里每一个更新方法都是在构造输出层的结构
 * 然后调用一连串的二分类器，将误差累计
 */
real Model::hierarchicalSoftmax(int32_t target, real lr)
{
  real loss = 0.0;
  grad_.zero();
  const std::vector<bool> &binaryCode = codes[target];
  const std::vector<int32_t> &pathToRoot = paths[target];
  // 目标单词的huffman编码就是每一个二分类器的label
  for (int32_t i = 0; i < pathToRoot.size(); i++)
  {
    loss += binaryLogistic(pathToRoot[i], binaryCode[i], lr);
  }
  return loss;
}

void Model::computeOutputSoftmax(Vector &hidden, Vector &output) const
{
  output.mul(*wo_, hidden);
  real max = output[0], z = 0.0;
  for (int32_t i = 0; i < osz_; i++)
  {
    max = std::max(output[i], max);
  }
  for (int32_t i = 0; i < osz_; i++)
  {
    output[i] = exp(output[i] - max);
    z += output[i];
  }
  for (int32_t i = 0; i < osz_; i++)
  {
    output[i] /= z;
  }
}

void Model::computeOutputSoftmax()
{
  computeOutputSoftmax(hidden_, output_);
}

real Model::softmax(int32_t target, real lr)
{
  grad_.zero();
  computeOutputSoftmax();
  for (int32_t i = 0; i < osz_; i++)
  {
    real label = (i == target) ? 1.0 : 0.0;
    real alpha = lr * (label - output_[i]);
    grad_.addRow(*wo_, i, alpha);
    wo_->addRow(hidden_, i, alpha);
  }
  return -log(output_[target]);
}

void Model::computeHidden(const std::vector<int32_t> &input, Vector &hidden) const
{
  assert(hidden.size() == hsz_);
  hidden.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it)
  {
    hidden.addRow(*wi_, *it);
  }
  hidden.mul(1.0 / input.size());
}

bool Model::comparePairs(const std::pair<real, int32_t> &l,
                         const std::pair<real, int32_t> &r)
{
  return l.first > r.first;
}

/**
 * 用于给输入数据打上1~K个类标签，并且输出各个类对应的概率值。
 * 对于Hierarchical softmax，需要遍历huafman 树才能找到top-k的结果
 * 对于普通的softmax（包括负采样和softmax的输出）需要遍历结果数组，找到top-k
 */ 
void Model::predict(const std::vector<int32_t> &input, int32_t k,
                    std::vector<std::pair<real, int32_t>> &heap,
                    Vector &hidden, Vector &output) const
{
  assert(k > 0);
  heap.reserve(k + 1);
  computeHidden(input, hidden);
  if (args_->loss == loss_name::hs)
  {
    dfs(k, 2 * osz_ - 2, 0.0, heap, hidden);
  }
  else
  {
    findKBest(k, heap, hidden, output);
  }
  std::sort_heap(heap.begin(), heap.end(), comparePairs);
}

void Model::predict(const std::vector<int32_t> &input, int32_t k,
                    std::vector<std::pair<real, int32_t>> &heap)
{
  predict(input, k, heap, hidden_, output_);
}

void Model::findKBest(int32_t k, std::vector<std::pair<real, int32_t>> &heap,
                      Vector &hidden, Vector &output) const
{
  computeOutputSoftmax(hidden, output);
  for (int32_t i = 0; i < osz_; i++)
  {
    if (heap.size() == k && log(output[i]) < heap.front().first)
    {
      continue;
    }
    heap.push_back(std::make_pair(log(output[i]), i));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k)
    {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
  }
}

void Model::dfs(int32_t k, int32_t node, real score,
                std::vector<std::pair<real, int32_t>> &heap,
                Vector &hidden) const
{
  if (heap.size() == k && score < heap.front().first)
  {
    return;
  }

  if (tree[node].left == -1 && tree[node].right == -1)
  {
    heap.push_back(std::make_pair(score, node));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k)
    {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
    return;
  }

  real f = sigmoid(wo_->dotRow(hidden, node - osz_));
  dfs(k, tree[node].left, score + log(1.0 - f), heap, hidden);
  dfs(k, tree[node].right, score + log(f), heap, hidden);
}
/**
 * 模型更新参数
 * input  :数组。其中每一个元素是dict里的ID。分类问题代表输入的文本，word2vec代表词的上下文
 * target :标签。分类问题就是类的标签，word2vec就是预测词的ID
 * lr     :参数的跟新速率
 */ 
void Model::update(const std::vector<int32_t> &input, int32_t target, real lr)
{
  assert(target >= 0);
  assert(target < osz_);
  if (input.size() == 0)
    return;
  // 输入层 -> 隐藏层 （只是将输入层的输入加权平均了）
  computeHidden(input, hidden_);
  // 根据输出层的不同，调用不同的求解函数
  // 1. Negative Sampling
  // 2. Hierarchical Softmax
  // 3. Softmax
  if (args_->loss == loss_name::ns)
  {
    loss_ += negativeSampling(target, lr);
  }
  else if (args_->loss == loss_name::hs)
  {
    loss_ += hierarchicalSoftmax(target, lr);
  }
  else
  {
    loss_ += softmax(target, lr);
  }
  // 样本数目加1
  nexamples_ += 1;

  if (args_->model == model_name::sup)
  {
    grad_.mul(1.0 / input.size());
  }
  // 词向量是在走完了输出层所有的二分类器之后才更新参数的
  // 输出层每一个二分类的参数是在上面调用的不同的求解函数的时候及时更新的
  for (auto it = input.cbegin(); it != input.cend(); ++it)
  {
    wi_->addRow(grad_, *it, 1.0);
  }
}

void Model::setTargetCounts(const std::vector<int64_t> &counts)
{
  assert(counts.size() == osz_);
  if (args_->loss == loss_name::ns)
  {
    initTableNegatives(counts);
  }
  if (args_->loss == loss_name::hs)
  {
    buildTree(counts);
  }
}

void Model::initTableNegatives(const std::vector<int64_t> &counts)
{
  real z = 0.0;
  for (size_t i = 0; i < counts.size(); i++)
  {
    z += pow(counts[i], 0.5);
  }
  for (size_t i = 0; i < counts.size(); i++)
  {
    real c = pow(counts[i], 0.5);
    for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++)
    {
      negatives.push_back(i);
    }
  }
  std::shuffle(negatives.begin(), negatives.end(), rng);
}

int32_t Model::getNegative(int32_t target)
{
  int32_t negative;
  do
  {
    negative = negatives[negpos];
    negpos = (negpos + 1) % negatives.size();
  } while (target == negative);
  return negative;
}

void Model::buildTree(const std::vector<int64_t> &counts)
{
  tree.resize(2 * osz_ - 1);
  for (int32_t i = 0; i < 2 * osz_ - 1; i++)
  {
    tree[i].parent = -1;
    tree[i].left = -1;
    tree[i].right = -1;
    tree[i].count = 1e15;
    tree[i].binary = false;
  }
  for (int32_t i = 0; i < osz_; i++)
  {
    tree[i].count = counts[i];
  }
  int32_t leaf = osz_ - 1;
  int32_t node = osz_;
  for (int32_t i = osz_; i < 2 * osz_ - 1; i++)
  {
    int32_t mini[2];
    for (int32_t j = 0; j < 2; j++)
    {
      if (leaf >= 0 && tree[leaf].count < tree[node].count)
      {
        mini[j] = leaf--;
      }
      else
      {
        mini[j] = node++;
      }
    }
    tree[i].left = mini[0];
    tree[i].right = mini[1];
    tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
    tree[mini[0]].parent = i;
    tree[mini[1]].parent = i;
    tree[mini[1]].binary = true;
  }
  for (int32_t i = 0; i < osz_; i++)
  {
    std::vector<int32_t> path;
    std::vector<bool> code;
    int32_t j = i;
    while (tree[j].parent != -1)
    {
      path.push_back(tree[j].parent - osz_);
      code.push_back(tree[j].binary);
      j = tree[j].parent;
    }
    paths.push_back(path);
    codes.push_back(code);
  }
}

real Model::getLoss() const
{
  return loss_ / nexamples_;
}

void Model::initSigmoid()
{
  t_sigmoid = new real[SIGMOID_TABLE_SIZE + 1];
  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++)
  {
    real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
    t_sigmoid[i] = 1.0 / (1.0 + std::exp(-x));
  }
}

void Model::initLog()
{
  t_log = new real[LOG_TABLE_SIZE + 1];
  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++)
  {
    real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
    t_log[i] = std::log(x);
  }
}

real Model::log(real x) const
{
  if (x > 1.0)
  {
    return 0.0;
  }
  int i = int(x * LOG_TABLE_SIZE);
  return t_log[i];
}

real Model::sigmoid(real x) const
{
  if (x < -MAX_SIGMOID)
  {
    return 0.0;
  }
  else if (x > MAX_SIGMOID)
  {
    return 1.0;
  }
  else
  {
    int i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid[i];
  }
}
}
