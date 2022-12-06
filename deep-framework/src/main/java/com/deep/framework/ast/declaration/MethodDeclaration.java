package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.TypeParametersExpression;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;
import com.deep.framework.ast.type.Type;
import lombok.Data;

import java.util.List;

@Data
public class MethodDeclaration extends Declaration {
    private List<TokenType> modifiers;
    private Type returnValue;
    private Name name;
    private Expression parameters;
    private BlockStatement body;

    public MethodDeclaration(Node prarent, List<TokenType> modifiers, Type returnValue, Name name, Expression parameters, BlockStatement body) {
        super(prarent);
        this.modifiers = modifiers;
        this.returnValue = returnValue;
        this.name = name;
        this.parameters = parameters;
        this.body = body;

        this.returnValue.setPrarent(this);
        this.name.setPrarent(this);
        this.parameters.setPrarent(this);
        this.body.setPrarent(this);

        getChildrens().addAll(returnValue, name, parameters, body);
    }

    public static void parser(Node node) {
        if (!(node.getPrarent() instanceof TypeDeclaration)) return;
        if (node instanceof MethodDeclaration) return;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            if (b instanceof BlockStatement) {
                CallableDeclaration.parser(a);
                Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                    if (n instanceof CallableDeclaration o) {
                        List<TokenType> modifiers = a.getMethodModifiers();
                        MethodDeclaration declare = new MethodDeclaration(node.getPrarent(), modifiers, Type.getType(m), o.getName(), o.getParameters(), (BlockStatement) b);
                        TypeParametersExpression.parser(declare);
                        node.replaceAndRemove(a, declare, b);
                    }
                });
            }
        });
    }
}