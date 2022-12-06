package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.TypeParametersExpression;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;
import lombok.Data;

import java.util.List;

@Data
public class ConstructorDeclaration extends Declaration {
    private List<TokenType> modifiers;
    private Name name;
    private Expression parameters;
    private BlockStatement body;

    public ConstructorDeclaration(Node prarent, List<TokenType> modifiers, Name name, Expression parameters, BlockStatement body) {
        super(prarent);
        this.modifiers = modifiers;
        this.name = name;
        this.parameters = parameters;
        this.body = body;

        this.name.setPrarent(this);
        this.parameters.setPrarent(this);
        this.body.setPrarent(this);

        getChildrens().addAll(name, parameters, body);
    }

    public static void parser(Node node) {
        if (!(node.getPrarent() instanceof TypeDeclaration)) return;
        if (node instanceof ConstructorDeclaration) return;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            if (b instanceof BlockStatement) {
                CallableDeclaration.parser(a);
                Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                    if (n instanceof CallableDeclaration o) {
                        List<TokenType> modifiers = a.getMethodModifiers();
                        if (a.isFirst(n)) {
                            ConstructorDeclaration declare = new ConstructorDeclaration(node.getPrarent(), modifiers, o.getName(), o.getParameters(), (BlockStatement) b);
                            TypeParametersExpression.parser(declare);
                            node.replaceAndRemove(a, declare, b);
                        }
                    }
                });
            }
        });
    }
}