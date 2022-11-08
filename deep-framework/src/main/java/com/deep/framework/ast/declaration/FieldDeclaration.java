package com.deep.framework.ast.declaration;

import com.deep.framework.ast.statement.BlockStatement;
import com.deep.framework.ast.type.Type;

public class FieldDeclaration extends Declaration {
    private String name;
    private String modifier;
    private Type type;
    private BlockStatement body;
}
