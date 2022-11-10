package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;
import lombok.Data;

import java.util.List;

@Data
public class ClassOrInterfaceDeclaration extends Declaration {
    private List<Object> extendedTypes;
    private List<Object> implementedTypes;

    private List<TokenType> modifiers;
    private Name name;
    private ParametersExpression parameters;
    private BlockStatement body;

    public ClassOrInterfaceDeclaration(Node prarent) {super(prarent);}
}
