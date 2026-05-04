@echo off
cd D:\TenderExtractPro
echo Current commit:
git log --oneline -1
echo.
echo Amending commit...
git commit --amend -m "Merge pull request #1 from prathameshfuke/version-2

Fix: Preserve filtered components in scoring logic

Co-authored-by: Gaurav Varu <gauravvaru2005@gmail.com>"
echo.
echo Pushing to main...
git push origin main --force-with-lease
echo.
echo Done!
pause
