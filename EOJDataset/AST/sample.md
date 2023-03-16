```c
#include<cstdio.h>

int main(){
    int x = 0;
    x = x + 1;
    printf("hello code %d",x);

    return 0;
}
```



```
ranslation_unit
preproc_include
#include
#include
<cstdio.h>
<cstdio.h>
preproc_include
function_definition
int
int
function_declarator
main
main
parameter_list
parameter_list
function_declarator
compound_statement
declaration
int
int
init_declarator
x
x
=
=
0
0
init_declarator
declaration
expression_statement
assignment_expression
x
x
=
=
binary_expression
x
x
+
+
1
1
binary_expression
assignment_expression
expression_statement
expression_statement
call_expression
printf
printf
argument_list
string_literal
string_literal
x
x
argument_list
call_expression
expression_statement
return_statement
return
return
0
0
return_statement
compound_statement
function_definition
translation_unit
```