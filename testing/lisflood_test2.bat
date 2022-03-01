:: carryout lisflood standard tests - ms-dos batch file
:: Mark Trigg 11/8/2008
::
:: Note this batch file will automatically pick up new 
:: test directories and parameter files - no need to update


@echo off

:: check if user wants to clean results directories or just run tests
IF "%1"=="" GOTO _TESTING
IF /I "%1"=="clean" GOTO :_CLEAN
echo ### WARNING Argument "%1" not recognised - exiting
GOTO _DONE


:_TESTING

:: setup header of test results output file

echo Lisflood standard tests > test_results.txt
echo Comparison between >> test_results.txt
lisflood_win_old.exe -version >> test_results.txt
echo and >> test_results.txt
lisflood_win.exe -version >> test_results.txt

:: loop through test Directories

FOR /F "usebackq tokens=*" %%d IN (`DIR /B /ON /AD T*`) DO (

  :: record dir info in test output file
  echo Processing directory %%d
  echo ######## Test directory ######## >> test_results.txt
  echo ================================ >> test_results.txt
  echo %%d >> test_results.txt
  echo ================================ >> test_results.txt
  echo - >> test_results.txt

  :: go into dir and record name
  cd %%d

  :: find all the parameter files
  for %%f in (*.par) do (

    :: record par file info in test output file
    echo. Found the file: %%f
    echo - >> ..\test_results.txt
    echo ### Parameter File %%f >> ..\test_results.txt
    echo - >> ..\test_results.txt

if exist old_%%f_results rd /s /q old_%%f_results
if exist new_%%f_results rd /s /q new_%%f_results

    :: run old and new lisflood on each par file
    ..\lisflood_win_old.exe -dir old_%%f_results %%f
    ..\lisflood_win.exe -dir new_%%f_results %%f

    :: compare model results and record output
    fc old_%%f_results\* new_%%f_results\* >> ..\test_results.txt
  )
  cd ..
)

echo ================================ >> test_results.txt
echo ==========END OF TESTS========== >> test_results.txt
echo ================================ >> test_results.txt

GOTO _DONE



:_CLEAN

echo CLEANING
:: loop through test Directories
FOR /F "usebackq tokens=*" %%d IN (`DIR /B /ON /AD T*`) DO (
  :: go into dir
  cd %%d

  :: find all results directories and delete them and their contents
  FOR /F "usebackq tokens=*" %%t IN (`DIR /B /ON /AD *results`) DO (
    RMDIR /S /Q %%t
  )

  cd ..
)
del /Q test_results.txt



:_DONE
echo FINISHED
