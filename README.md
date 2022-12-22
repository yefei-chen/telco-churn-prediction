# 电信客户流失分析与预测

## 项目背景
随着市场饱和度的上升，电信运营商的竞争也越来越激烈，再加上高昂的客户获取成本，流失分析就变得非常关键。流失率是一种指标，用于描述取消或未续订公司套餐的客户数量。对于客户流失率而言，每增加5%，利润就可能随之降低25%-85%。因此，如何减少电信客户流失的分析与预测至关重要。基于从客户流失分析中获得的信息，电信公司可以制定战略、瞄准细分市场，提高所提供服务的质量以改善客户体验，从而培养客户的信任度。

## 项目目标
本文将基于IBM的电信客户数据集（数据来源：[Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)），进行探索性数据分析以洞察客户特征与流失的关系，构建特征工程并运用分类机器学习算法建立模型，尝试找到合适的模型预测流失客户，从而为运营商的客户服务部门提供决策依据。

## 理解数据
数据共计7043行，21列（字段）。每行代表一个客户，每列包含一个唯一客户属性，其中，第一列是客户ID，最后一列“Churn“表示用户是否流失。各字段的具体含义如下：

<table border="1" cellpadding="1" cellspacing="1" style="width:500px">
	<tbody>
		<tr>
			<td style="text-align:center; width:36px">序号</td>
			<td style="text-align:center; width:89px">字段名</td>
			<td style="text-align:center">数据类型</td>
			<td style="text-align:center">字段描述</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">1</td>
			<td style="text-align:center; width:89px">customerID</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">客户ID</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">2</td>
			<td style="text-align:center; width:89px">gender</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">性别（男，女）</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">3</td>
			<td style="text-align:center; width:89px">SeniorCitizen</td>
			<td style="text-align:center">Integer</td>
			<td style="text-align:center">客户是否为老年人（是为1，不是为0）</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">4</td>
			<td style="text-align:center; width:89px">Partner</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">客户是否有伴侣（是，否）</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">5</td>
			<td style="text-align:center; width:89px">Dependents</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">客户是否有家属（是，否）</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">6</td>
			<td style="text-align:center; width:89px">tenure</td>
			<td style="text-align:center">Integer</td>
			<td style="text-align:center">客户已使用月数</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">7</td>
			<td style="text-align:center; width:89px">PhoneService</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">客户是否使用电话服务（是，否）</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">8</td>
			<td style="text-align:center; width:89px">MultipleLines</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">客户是否有多条线路（是，否，没有电话服务）</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">9</td>
			<td style="text-align:center; width:89px">InternetService</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">客户的互联网服务提供商（DSL，光纤，否）</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">10</td>
			<td style="text-align:center; width:89px">OnlineSecurity</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">客户是否具有在线安全性（是，否，没有互联网服务）</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">11</td>
			<td style="text-align:center; width:89px">OnlineBackup</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">客户是否具有在线备份（是，否，没有互联网服务）</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">12</td>
			<td style="text-align:center; width:89px">DeviceProtection</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">客户是否具有设备保护（是，否，没有互联网服务）</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">13</td>
			<td style="text-align:center; width:89px">TechSupport</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">客户是否具有技术支持（是，否，没有互联网服务）</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">14</td>
			<td style="text-align:center; width:89px">StreamingTV</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">客户是否具有流媒体电视（是，否，没有互联网服务）</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">15</td>
			<td style="text-align:center; width:89px">StreamingMovies</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">客户是否具有流媒体电影（是，否，没有互联网服务）</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">16</td>
			<td style="text-align:center; width:89px">Contract</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">客户的合同期限（每月，一年，两年）</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">17</td>
			<td style="text-align:center; width:89px">PaperlessBilling</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">客户是否具有无纸化账单（是，否）</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">18</td>
			<td style="text-align:center; width:89px">PaymentMethod</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">客户的付款方式（电子支票，邮寄支票，银行转账（自动），信用卡（自动））</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">19</td>
			<td style="text-align:center; width:89px">MonthlyCharges</td>
			<td style="text-align:center">Integer</td>
			<td style="text-align:center">每月向客户收取的金额</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">20</td>
			<td style="text-align:center; width:89px">TotalCharges</td>
			<td style="text-align:center">Integer</td>
			<td style="text-align:center">向客户收取的总金额</td>
		</tr>
		<tr>
			<td style="text-align:center; width:36px">21</td>
			<td style="text-align:center; width:89px">Churn</td>
			<td style="text-align:center">String</td>
			<td style="text-align:center">客户是否流失（是，否）</td>
		</tr>
	</tbody>
</table>

<p style="text-align:center">&nbsp;</p>

## 数据预处理
通过对数据的概览，删除空白值，缺失值补全和字段类型转换等手段对原始数据进行预处理，以便后续分析。详见notebook。

## 探索性数据分析
分别基于数值特征和分类特征进行Exploratory data analysis，洞察数据中蕴含的深层信息，构建流失用户画像，并为特征工程提供数据直觉。详见notebook。

将所有分类特征划分为用户维度、服务维度、合同维度，并分别从三个维度进行探索：

用户维度：gender, SeniorCitize, Partner, Dependents
<p>小结：</p>

<ul>
	<li>流失与性别无关</li>
	<li>老年客户占比较小，但流失率更高</li>
	<li>拥有伴侣或家属的客户，流失率更低</li>
</ul>

服务维度：PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
<p>小结：</p>

<ul>
	<li>大多数客户开通了电话服务，但是开通和未开通的流失比例相当；开通电话服务的人群中，接近半数的人群开通了多条线路，开通和未开通的流失比例也相差不大，可见电话服务（无论是单线还是多线）对客户整体流失影响较小</li>
	<li>大多数客户选择开通网络服务，其中选择光纤的人数比选择DSL的人数更多，然而，光纤客户的流失占比却更高；通过查看网络服务子服务流失情况可以发现，如果开通了安全、备份、保护、技术支持这些绑定服务流失率会降低；但流媒体电视和流媒体电影这两项子服务似乎没有前者留存效果好</li>
</ul>

合同维度：Contract, PaperlessBilling, PaymentMethod
<p>小结：</p>

<ul>
	<li>开通按月服务的客户占绝大多数，可见大部分客户抱有一种试用的心理，相应的，按月服务的流失率也最高，客户可能在试用期结束后转向了其他服务商</li>
	<li>很多客户开通了无纸化账单服务，但是流失率也随之提升</li>
	<li>支付方式中，使用电子支票的客户数目最多，流失率也最高，推测该方式使用体验一半</li>
</ul>

并针对数值特征进行探索：
<p>小结：</p>

<ul>
	<li>在使用时长小于20个月时，似乎不同级别的月支出对于客户流失影响不大；随着使用时长继续增加，月支出较高的这部分客户开始流失</li>
	<li>使用时长和总支出呈正相关，同样使用时长的情况下，似乎总支出越高，客户流失的可能性越大</li>
	<li>月支出和总支出也呈正相关，这是符合常理的</li>
</ul>

总结：
通过探索性数据分析，可以得到较高流失率的客户特征，具有这些特征的客户群体需要采取针对性的运营策略，增加客户粘性，延长其生命周期价值。

<img width="505" alt="image" src="https://user-images.githubusercontent.com/49276153/209122877-ebe6632d-73ba-405f-9d6e-fff2771809d3.png">

用户维度：可针对老年人、无伴侣、无亲属的群体推出定制化服务，例如亲子套餐，加强其与社会关系的关联度，同时有可能发展更多的客户
服务维度：可针对新注册用户，在试用期间多推出新用户活动，例如前半年赠送话费代金券，以增强用户粘性，渡过用户流失高峰期。针对光纤用户（尤其是开通电视电影服务），可以重点提升服务体验，例如网速升级，媒体内容增多等。另一方面可以提供包月服务。
合同维度：针对按月合同用户，针对性地推送年包折扣活动，将月用户转换为年用户，提高用户留存。优化电子支票支付方式，或是建议用户转向其他支付。

## 特征工程
将数据进行归一化，并分别基于卡方检验和ANOVA检验的方式计算特征相关系数，选取典型分类特征和数值特征。

## 建模与预测

构建机器学习模型，对流失用户进行预测。本文用了决策树、随机森林、GBDT模型、Xgboost模型以及这四个模型的Stacking融合模型分别进行预测，预测效果见表：

|模型  |  CV score| ROC AUC score| F1 score
|--|--|--|--|
|  决策树|84.80%  | 77.02% |    0.77
| 随机森林 | 85.82% |78.97%  |  0.79  
|  GBDT模型|90.44%  |83.01%  | 0.83   
|  Xgboost模型|90.47%  | 83.15% | 0.83   
| Stacking模型 | 91.20% | 83.77% | 0.84   

## 结论与建议
本文首先对电信客户流失数据进行清洗，然后分别从分类特征和数值特征的角度进行探索性分析，得到流失用户的基本画像，为客服部门针对性运营提供参考意见。

同时洞察数据中的深层次信息，并基于这些印象，选择典型特征，构建机器学习模型，对流失用户进行预测。本文用了决策树、随机森林、GBDT模型、Xgboost模型以及这四个模型的Stacking融合模型分别进行预测，预测准确程度依次提升。客服部门可以基于预测模型，构建高流失用户列表，并深入优化服务（例如针对这一部分人群进行访谈）。
