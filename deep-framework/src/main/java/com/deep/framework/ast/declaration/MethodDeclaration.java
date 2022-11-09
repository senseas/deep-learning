package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;
import com.deep.framework.ast.type.Type;
import lombok.Data;

@Data
public class MethodDeclaration extends Declaration {
    public TokenType modifier;
    public Type returnValue;
    public ParametersExpression parameters;
    public Name name;
    private BlockStatement body;

    public MethodDeclaration(Node prarent) {
        super(prarent);
    }
}
