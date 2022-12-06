package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.declaration.*;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;
import lombok.Data;

import java.util.List;

@Data
public class ObjectCreationExpression extends TypeDeclaration {
    private Name name;
    private Expression parameters;
    private BlockStatement body;
    private static ObjectCreationExpression expression;

    public ObjectCreationExpression(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            CallableDeclaration.parser(a);
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m.equals(TokenType.NEW) && n instanceof CallableDeclaration o) {
                    expression = new ObjectCreationExpression(node);
                    expression.setParameters(o.getParameters());
                    expression.setName(o.getName());
                    a.replace(m, expression);
                    a.getChildrens().remove(n);

                    if (b instanceof BlockStatement) {
                        b.setPrarent(expression);
                        expression.setBody((BlockStatement) b);
                        expression.getChildrens().addAll(List.of(m, n, b));

                        ConstructorDeclaration.parser(b);
                        MethodDeclaration.parser(b);
                        FieldDeclaration.parser(b);

                        node.getChildrens().remove(b);
                        list.remove(b);
                    }
                }
            });
        });
    }

    @Override
    public String toString() {
        return "new ".concat(name.toString()).concat("(").concat(parameters.toString()).concat(")");
    }
}