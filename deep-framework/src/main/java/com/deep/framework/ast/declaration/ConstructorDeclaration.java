package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import com.deep.framework.ast.statement.BlockStatement;

public class ConstructorDeclaration extends Declaration {
    private String modifier;
    private Name name;
    private ParametersExpression parameters;
    private BlockStatement body;

    public ConstructorDeclaration(Node prarent) {super(prarent);}
}