# Developer notes


## Markdown documentation

Think to document the firmware, describe its behaviour, with the markdown the repository (the entry point is README.md).

The markdown documentation can be checked on the GIT repository website (see next chapter).
It can also be checked and viewed offline by using the **grip** tool.
It may be installed as follows by the root user:

```
pip install grip
```

If figures are used in the markdown documentation, please store the source file as well as the picture (jpg, png), so that it can be amended by others.
A dedicated **fig** directory may be used.

Use preferably open source software (libreoffice draw) so that modifications can be done on any computing plateform by others.
Along the libreoffice figure, you may store a Makefile to automate the picture generation, see the Makefile stored in the **fig** directory.

Additionnaly, a pdf file can be generated for each markdown file with **pandoc** tool:

```
pandoc README.md -t latex -o output.pdf
```

## GIT repository

The code is hosted on the following git [repository](https://gitlab.lam.fr/KISS/kidsdata). 
The code can be cloned:
- Via password authentification

```
git clone https://gitlab.lam.fr/KISS/kidsdata.git
```

- Via SSH keys

```
git clone git@gitlab.lam.fr:KISS/kidsdata.git
```

To clone directly a branch and not the default one (here develop branch):

```
git clone git@gitlab.lam.fr:KISS/kidsdata.git -b develop
```

The repository host several branches:
- **master**: the default one, it host the officilally released versions. It is a protected branch.
- **develop**: Used by developpers, it should host a working version (not guaranteed!). It is a protected branch.
- The other branches corresponds to feature enhancement, developper playground and should not be considered by any user.


## Recommended usage

Any developper willing to contribute should create a branch and work in this branch, unless he or she is invited to work in a shared branch by another developper.

All consolidated work should be merged into the **develop** branch, the merge in **master** branch should occur only for official release.

The recommended procedure for merging with the **develop** branch is to rebase your branch from the develop branch, this will keep a linear history.
Unless you feel that the merging should appear clearly on the history log (in the decorated log and via a dedicated merge message).

Whatever the choice, you should check that your branch is still working before merging with develop.
The recipe would be the following
1. Finishing your work and commit your work in your branch (avoid the use of git commit -a or carefully review the added files and abort if innapropriate)
2. Change to develop branch and do a git pull to retrieve the latest contribution
3. Change to your branch and then either rebase (prefered) or merge your branch with develop
4. Check that everything is fine (if it takes times, you may have to go through previous step again)
	- If you rebased, clean your local commit history log.
5. Go to the develop branch (do a pull again to check that no changes occured, if it does, go to step 3)
6. Merge your branch in develop and push it to remote



### Versionning and documentation

Think to use  meaningfull commit messages. As long as your work is not pushed to the remote, you can always rewrite your local history, either by amending your latest commit or by rewriting your history (interactive rebase).
After rebasing your branch  (not merging!), you may want to modify your latest commits to fix an issue, you may use the rebase  and/or commit ammend on your rehashed revision.
Beware, that if you modify already pushed commits, you may not be allowed to push your design again!

Indeed, you can only force to push modified history on unprotected branches. Do that if you are absolutely certain that no one else is using your branch and that the history part you modify has not already been merged with another branch.
So it can be done, **but with caution**.

So, please look at the decorated graph before modifying your history, it will show you the latest pushed/pulled.

When entering issue, merge request or commit messages, remember that they can be referenced with the following:
- Issues with '#' in front of the issue number
- Merge requests with '!' in front of the issue number
- Commit, just by indicating the hash number



## Some useful commands

To create a local branch :

```
git branch mybranch
```

To create a local branch and immediatly change to this newly created branch (preferred)

```
git checkout -b mybranch
```

To share your local branch (send it to remote):

```
git push origin mybranch
```

To see existing remote branches:

```
git branch -r
```

To track a remote branch:

```
git checkout -t remotename/branch
```

To rebase the branch you are in with develop
```
git rebase develop
```

To merge the branch you are in with develop
```
git merge develop
```

To merge your branch with develop (once everything is OK):
```
git merge mybranch
```

To modify your latest commit (add missing files or change the message):
```
git commit --amend
```

To clean your (local) history (squash, rewording, ...) :
```
git rebase -i
```

Show decorated history
```
git log --graph --abbrev-commit --decorate --date=relative --all
```


## Some useful git preferences

```
git config --global color.diff auto
git config --global color.status auto
git config --global color.branch auto
git config --global diff.tool meld
git config --global core.editor vim
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)%Creset' --abbrev-commit"
```

To show your configuration:

```
git config --list
```

