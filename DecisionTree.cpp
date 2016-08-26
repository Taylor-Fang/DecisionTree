#include "DecisionTree.h"

int bestIndex;//最佳分裂属性

void DecisionTree::count(DataTable & samples, pCountCollection &sampleCount, IndexCollection & attrIndex,ClassCollection & classArray,DividerSet & dividerArray)
{
	sampleCount = new CountCollection [attrIndex.size()];
	int colCount = (int)attrIndex.size();//属性数(列数)
	int valueNum = 2;//因为是连续属性，只有两种结果
	int rowNum = (int)classArray.size();//样本行数

	//构造统计结构并初始化
	for(int col = 0;col < colCount; col++)
	{
		for(int i = 0; i < valueNum; i++)
		{
			//sampleCount存储统计结果
			//每个属性存储4个统计结果
			/******************************************
			 *           * <最优分割值 * >=最优分割值 *
			 ******************************************
			 * [0]类别-1 *             *              *
			 ******************************************
			 * [1]类别1  *             *              *
			 ******************************************/
			sampleCount[col].push_back(new int [2]);//2表示只有两种类别
			for(int j = 0;j < 2;j++)
				sampleCount[col][i][j] = 0;//初始化
		}
	}

	//统计
	for(int col = 0;col < colCount; col++)
	{
		double bestDivider = 0;//连续属性的最优分割值
		bestDivider = discrete(rowNum,col,classArray,samples,attrIndex[col]);//获取col列最优分割值
		dividerArray.push_back(bestDivider);

		for(int rowIndex = 0;rowIndex < rowNum;rowIndex++)
		{
			//flag == 0表示小于分割值 flag == 1表示大于等于分割值
			int flag = 1;
			if(samples[rowIndex][col] < bestDivider)
				flag = 0;

			//m == 0表示类别-1,m == 1表示类别1
			int m = 0;
			if(classArray[rowIndex] == 1)
				m = 1;
			sampleCount[col][flag][m]++;
		}
	}
}

//获取col列属性的最优分割值
double DecisionTree::discrete(int rowNum, int colIndex, ClassCollection & classArray, DataTable & samples, int truecol)
{
	//存储本列属性的数据值
	double *attributeValue = new double [rowNum];

	//存储属性值所对应的类别值
	int *attributeLabel = new int [rowNum];
	for(int row = 0;row < rowNum;row++)
	{
		attributeValue[row] = samples[row][colIndex];
		attributeLabel[row] = classArray[row];
	}

	//排序（冒泡排序法，需要改进）
	for(int row = 0; row < rowNum - 1; row++)
	{
		for(int j = rowNum - 1;j > row; j--)
		{
			if(attributeValue[j] < attributeValue[j-1])
			{
				double temp1 = attributeValue[j];
				attributeValue[j] = attributeValue[j-1];
				attributeValue[j-1] = temp1;
				int temp2 = attributeLabel[j];
				attributeLabel[j] = attributeLabel[j-1];
				attributeLabel[j-1] = temp2;
			}
		}
	}
	//little存储小于分割值的每种类别数量，large反之
	//little[0]存储类别为-1的数量，little[1]存储类别为1的数量，large同理
	int little[2],large[2];
	double bestDivider = 0.0;//bestDivider最优分割值
	double maxGainRatio = -1000;//maxGainRatio最大增益率

	for(int row = 0;row < rowNum-1;row++)
	{
		if(attributeLabel[row] != attributeLabel[row+1])
		{
			for(int i = 0;i < 2; i++)
			{
				little[i] = 0;
				large[i] = 0;
			}
		
			for(int j = 0;j < row+1; j++)
			{
				if(attributeLabel[j] == -1)
					little[0]++;
				else
					little[1]++;
			}

			for(int j = row+1;j < rowNum; j++)
			{
				if(attributeLabel[j] == -1)
					large[0]++;
				else
					large[1]++;
			}
		}
		//获取最大的增益率
		double infoGainRatio = gainRatio(rowNum,little,large);

		if(infoGainRatio > maxGainRatio)
		{
			bestDivider = attributeValue[row+1];
			maxGainRatio = infoGainRatio;
		}
	}
	delete [] attributeValue;
	delete [] attributeLabel;

	return bestDivider;
}

//计算信息熵
//labelCount[0]表示类别-1的数目，labelCount[1]表示类别1的数目，allNum表示总数
double DecisionTree::entropy(int *labelCount)
{
	double iEntropy = 0.0;
	int allNum = labelCount[0]+labelCount[1];
	for(int i = 0; i < 2; i++)
	{
		double temp = ((double)labelCount[i])/allNum;
		if(temp != 0.0)
			iEntropy -= temp*(log(temp)/log(2.0));
	}

	return iEntropy;
}

//计算增益率
double DecisionTree::gainRatio(int allNum,int *little,int *large)
{
	double gain = 0.0;
	double splitInfo = 0.0;
	int labelCount[2] = {0,0};
	labelCount[0] = little[0] + large[0];
	labelCount[1] = little[1] + large[1];
	
	double sampleEntropy = entropy(labelCount);
	double entropy1 = (double(little[0]+little[1]))/allNum*entropy(little);
	double entropy2 = (double(large[0]+large[1]))/allNum*entropy(large);
	gain = sampleEntropy - entropy1 - entropy2;

	double splitInfo1 = -(double(little[0]+little[1]))/allNum*(log(double(little[0]+little[1])/allNum)/log(2.0));
	double splitInfo2 = -(double(large[0]+large[1]))/allNum*(log(double(large[0]+large[1])/allNum)/log(2.0));
	splitInfo = splitInfo1 + splitInfo2;

	return (gain/splitInfo);
}

//选择最佳分裂属性
int DecisionTree::chooseBestAttribute(IndexCollection attrIndex, pCountCollection sampleCount)
{
	int bestIndex = 0;
	double maxGainRatio = -100.0;
	int allNum = sampleCount[0][0][0]+sampleCount[0][0][1]+sampleCount[0][1][0]+sampleCount[0][1][1];

	for(int colIndex = 0;colIndex < (int)attrIndex.size();colIndex++)
	{
		double gainR = gainRatio(allNum,sampleCount[colIndex][0],sampleCount[colIndex][1]);
		if(gainR > maxGainRatio)
		{
			maxGainRatio = gainR;
			bestIndex = colIndex;
		}
	}

	return bestIndex;
}

//判断样本中类别是否都一致
bool DecisionTree::isSameResultValue(ClassCollection & classArray)
{
	int temp = classArray[0];
	bool same = true;
	for(int row = 0;row < classArray.size();row++)
	{
		if(classArray[row] != temp)
		{
			same = false;
			break;
		}
	}

	return same;
}

//建立决策树
pNode DecisionTree::buildTree(DataTable & samples, IndexCollection attrIndex, ClassCollection & classArray)
{
	int rowCount = (int)samples.size();//样本数
	int colCount = (int)attrIndex.size();//属性数(不包含类别属性)

	//若样本为空则结束,需修改，应直接返回
	if(samples.size() == 0)
	{
		pNode leafNode = new Node ;
		leafNode->isLeaf = true;
		leafNode->attrIndex = -1;//没有分裂属性，令其为-1
		leafNode->popularClass = 0;//空
		leafNode->bestDivider = -1;
		leafNode->count[0] = 0;
		leafNode->count[1] = 0;
		leafNode->leftChild = nullptr;
		leafNode->rightChild = nullptr;

		return leafNode;
	}

	//如果样本属于同一类别
	if(isSameResultValue(classArray))
	{
		int temp = classArray[0];
		pNode leafNode = new Node;
		leafNode->isLeaf = true;
		leafNode->attrIndex = -1;//样本属于同一类别，因此不需要再分裂
		leafNode->popularClass = temp;
		leafNode->bestDivider = -1;//不需要分裂，所以也不需要分割值
		leafNode->count[0] = rowCount;
		leafNode->count[1] = 0;
		leafNode->leftChild = nullptr;
		leafNode->rightChild = nullptr;

		return leafNode;
	}

	pNode iNode = new Node;
	int labelNum = 2;//类别种类数
	int checkCount[2] = {0,0};
	for(int rowIndex = 0;rowIndex < rowCount;rowIndex++)
	{
		if(classArray[rowIndex] == -1)
			checkCount[0]++;
		else
			checkCount[1]++;
	}

	int maxCount = 0;
	int popularClass = 0;
	if(checkCount[0] > checkCount[1])
	{
		maxCount = checkCount[0];
		popularClass = -1;
	}
	else
	{
		maxCount = checkCount[1];
		popularClass = 1;
	}

	iNode->popularClass = popularClass;
	iNode->count[0] = maxCount;
	iNode->count[1] = rowCount - maxCount;

	//如果没有可分裂的条件属性
	if(attrIndex.size() == 0)
	{
		iNode->isLeaf = true;
		iNode->attrIndex = -1;//没有可分裂的属性
		iNode->bestDivider = -1;
		iNode->leftChild = nullptr;
		iNode->rightChild = nullptr;
		return iNode;
	}

	//递归，构建子树
	pCountCollection sampleCount;
	DividerSet dividerArray;
	count(samples,sampleCount,attrIndex,classArray,dividerArray);
	
	bestIndex = chooseBestAttribute(attrIndex,sampleCount);
	for(int colIndex = 0;colIndex < colCount;colIndex++)
		for(int i = 0;i < 2;i++)
			delete [] sampleCount[colIndex][i]; 
	delete [] sampleCount;

	int bestAttribute = attrIndex[bestIndex];
	
	iNode->isLeaf = false;
	iNode->attrIndex = bestAttribute;
	iNode->bestDivider = dividerArray[bestIndex];
	int valueN = 2;//连续属性，故分裂为两个子树
	iNode->leftChild = new Node;
	iNode->rightChild = new Node;
	DataTable *iSamples = new DataTable [valueN];
	ClassCollection *iclassArray = new ClassCollection [valueN];
	DataRow row;
	for(int rowIndex = 0;rowIndex < rowCount;rowIndex++)
	{
		int flag = 1;
		if(samples[rowIndex][bestIndex] < dividerArray[bestIndex])
			flag = 0;
		for(int colIndex = 0;colIndex < colCount;colIndex++)
		{
			if(colIndex != bestIndex)
				row.push_back(samples[rowIndex][colIndex]);
		}
		iSamples[flag].push_back(row);
		iclassArray[flag].push_back(classArray[rowIndex]);
		row.clear();
	}
	dividerArray.clear();

	IndexCollection newAttrIndex;
	for(int i = 0; i < (int)attrIndex.size(); i++)
	{
		if(i != bestIndex)
			newAttrIndex.push_back(attrIndex[i]);
	}
	iNode->leftChild = buildTree(iSamples[0], newAttrIndex,iclassArray[0]);
	iNode->rightChild = buildTree(iSamples[1], newAttrIndex,iclassArray[1]);
	
	delete [] iSamples;
	delete [] iclassArray;

	return iNode;
}

//统计子树叶子节点个数,中序遍历
int DecisionTree::leafCount(pNode & iNode)
{
	if(iNode == nullptr)
		return 0;
	if((iNode->leftChild == nullptr) && (iNode->rightChild == nullptr))
		return 1;

	return (leafCount(iNode->leftChild)+leafCount(iNode->rightChild));
}

//统计子树叶子节点中错误样本数
int DecisionTree::errCount(pNode &iNode)
{
	if(iNode == nullptr)
		return 0;
	if((iNode->leftChild == nullptr) && (iNode->rightChild == nullptr))
		return iNode->count[1];

	return (errCount(iNode->leftChild)+errCount(iNode->rightChild));
}

//悲观剪枝法(需要修改)
bool DecisionTree::postPrune(pNode &iNode)
{
	if(iNode->isLeaf)
		return false;
	int leafNum = 0;//统计叶子节点的数目
	int errNum = 0; //统计叶子节点中错误样本数
	leafNum = leafCount(iNode);
	errNum = errCount(iNode);
	double e = (double)(errNum+(double)leafNum/2)/(iNode->count[0]+iNode->count[1]);//误判率
	//E_subtree = 叶子节点中错误总和 + 0.5 * 叶子节点数
	double E_subtree = errNum + (double)leafNum/2;
	//E_node = 该节点的错误数 + 0.5
	double E_node = iNode->count[1] + 0.5;
	//标准差var = sqrt(N*e(1-e))
	double var = sqrt(E_subtree*(1-e));

	if((var+E_subtree) >= E_node)
	{
		Node * newNode = new Node;
		newNode->isLeaf = true;
		newNode->bestDivider = -1;
		newNode->attrIndex = -1;
		newNode->count[0] = iNode->count[0];
		newNode->count[1] = iNode->count[1];
		newNode->popularClass = iNode->popularClass;
		newNode->leftChild = nullptr;
		newNode->rightChild = nullptr;

		//删除原来的子树
		delete iNode;
		iNode = newNode;
		return true;
	}
	else
	{
		postPrune(iNode->leftChild);
		postPrune(iNode->rightChild);
	}
}

void DecisionTree::removeTree(pNode &head)
{
	int valueN = 2;
	if(head->isLeaf)
	{
		delete head;
		return;
	}
	removeTree(head->leftChild);
	removeTree(head->rightChild);

	delete head;
}

Result DecisionTree::test(pNode & treeNode,DataTable & testSet,ClassCollection & testClass)
{
	Result res;
	res.count = testSet.size();
	int correctCount = 0;

	//遍历测试表
	for(int row = 0;row < testSet.size();row++)
	{
		bool flag = todotest(testSet[row],treeNode,testClass[row]);
		if(flag == true)
			correctCount++;
	}

	res.rate = (double)correctCount/res.count;
	
	return res;
}

bool DecisionTree::todotest(DataRow & row,pNode & treeNode,int label)
{
	if(treeNode->isLeaf)
	{
		if(treeNode->popularClass == label)
			return true;
		else
			return false;
	}
	else
	{
		int attrIndex = treeNode->attrIndex;
		bool flag = false;
		if(row[attrIndex] < treeNode->bestDivider)
		    flag = todotest(row,treeNode->leftChild,label);
		else
			flag = todotest(row,treeNode->rightChild,label);
		return flag;
	}
}