package com.deep.framework.ast.declaration;

import com.deep.framework.ast.statement.BlockStatement;
import com.deep.framework.ast.type.Type;

public class MethodDeclaration extends Declaration {
    public String name;
    public String returnValue;
    public String modifier;
    public Type type;
    private BlockStatement body;
}
