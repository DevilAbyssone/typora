1.查看git配置信息 git config --list
2.查看git用户名 git config user.name
3.查看邮箱配置 git config user.email
4.全局配置用户名 git config --global user.name "name...
5.全局配置邮箱 git config --global user.email "eamil“
git.status 查看当前仓库状态 文件
git add filename 将文件加入暂存区 git add.将所有的文件都加入暂存区
git commit -m " 备注" 提交  只提交绿色的文件
git.log 查看日志
git reset 将绿色文件变成红色git reset commitID --hard不保存所有变更 --soft保留变更且变更内容属于Staged绿色状态 --mixed 保留变更且保留变更内容处于Modofied已修改红色状态
git reflog查找撤销那次的id 用reset又可以回退
git pull回到最新的commit
git checkpoint -b name template（默认当前分支 master
git checkout name 切换分支
git branch拿到所有分支 
可以切分和汇聚分支 https://blog.csdn.net/halaoda/article/details/78661334
git clone 拿项目

