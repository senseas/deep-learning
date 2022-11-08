package com.deep.framework.ast.declaration;

import com.deep.framework.ast.statement.BlockStatement;
import com.deep.framework.ast.statement.Statement;

import java.util.List;

public class ConstructorDeclaration extends Declaration {
    private String access;
    private String name;
    private List<Statement> agrments;

    private BlockStatement body;
}
