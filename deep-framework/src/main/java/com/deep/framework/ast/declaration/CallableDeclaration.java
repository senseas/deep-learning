package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.LambdaExpression;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import static com.deep.framework.ast.lexer.TokenType.SUPER;

@Data
public class CallableDeclaration extends Declaration {
    private Expression parameters;
    private Name name;
    private static CallableDeclaration declare;

    public CallableDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (node instanceof CallableDeclaration) return;
        if (node instanceof MethodDeclaration) return;
        LambdaExpression.parser(node);
        Stream.of(node.getChildrens()).reduce((list, m, n) -> {
            if (m instanceof Name && n instanceof ParametersExpression) {
                declare = new CallableDeclaration(node);
                declare.setName((Name) m);
                declare.setParameters((Expression) n);
                declare.getChildrens().add(n);
                node.replaceAndRemove(m, declare, n);
                list.remove(n);
            } else if (m instanceof Name && n instanceof LambdaExpression) {
                declare = new CallableDeclaration(node);
                declare.setName((Name) m);
                declare.setParameters((Expression) n);
                declare.getChildrens().add(n);
                node.replaceAndRemove(m, declare, n);
                list.remove(n);
            } else if (m.equals(SUPER) && n instanceof ParametersExpression) {
                declare = new CallableDeclaration(node);
                Name name = new Name(m.getTokenType());
                declare.setName(name);
                declare.setParameters((Expression) n);
                declare.getChildrens().add(n);
                node.replaceAndRemove(m, declare, n);
                list.remove(n);
            } else if (m.equals(SUPER) && n instanceof LambdaExpression) {
                declare = new CallableDeclaration(node);
                Name name = new Name(m.getTokenType());
                declare.setName(name);
                declare.setParameters((Expression) n);
                declare.getChildrens().add(n);
                node.replaceAndRemove(m, declare, n);
                list.remove(n);
            }
        });
    }

    @Override
    public String toString() {
        return name.toString().concat("(").concat(parameters.toString().concat(")"));
    }
}